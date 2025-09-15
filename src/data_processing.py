"""Data preprocessor module for cooling system analysis."""

import pandas as pd
import numpy as np
import re
from typing import Optional, Dict
from pathlib import Path

from constants import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TIME_COLUMN,
    COLUMN_MAPPING,
    PRIMARY_ENERGY_TARGETS,
    FAN_CONTROL_VARIABLES,
    COOLING_TOWER_CONTROL_VARIABLES,
    LOAD_RATE_PERCENTILE,
    OPERATION_THRESHOLD,
)


class DataPreprocessor:
    """Handle data cleaning and basic preprocessing for cooling system data."""

    def __init__(
        self, input_path: Path = RAW_DATA_PATH, output_path: Path = PROCESSED_DATA_PATH
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None

    def clean_column_names(self) -> None:
        """Clean column names by removing special characters and newlines."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Create a mapping for current columns to cleaned versions
        current_columns = self.df.columns.tolist()
        cleaned_mapping = {}

        for col in current_columns:
            # Remove special characters and newlines, but keep mapping if exists
            cleaned_col = re.sub(r"[\(\)\%\n\r]", "", str(col)).strip()
            cleaned_col = re.sub(
                r"\s+", " ", cleaned_col
            )  # Replace multiple spaces with single space

            # Use predefined mapping from config if available, otherwise use cleaned version
            if col in COLUMN_MAPPING:
                cleaned_mapping[col] = COLUMN_MAPPING[col]
            else:
                # Convert to snake_case for consistency
                snake_case_col = cleaned_col.lower().replace(" ", "_").replace("-", "_")
                snake_case_col = re.sub(r"[^a-z0-9_]", "", snake_case_col)
                cleaned_mapping[col] = snake_case_col

        # Apply the mapping
        self.df = self.df.rename(columns=cleaned_mapping)
        print(f"Column names cleaned and mapped to English")

    def clean_data_values(self) -> None:
        """Clean data values by removing special characters from string columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get string/object columns
        string_columns = self.df.select_dtypes(include=["object"]).columns

        for col in string_columns:
            if col != "time":  # Skip time column
                # Remove special characters from string values
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .apply(
                        lambda x: (
                            re.sub(r"[\(\)\%\n\r]", "", str(x)).strip()
                            if pd.notna(x)
                            else x
                        )
                    )
                )

                # Convert cleaned strings to numeric if possible
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    pass

        print("Data values cleaned of special characters")

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Raw data file not found: {self.input_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def clean_time_column(self) -> None:
        """Clean and format time column."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Handle the NaT feature column
        if "Feature (NaT)" in self.df.columns:
            self.df = self.df.astype({"Feature (NaT)": "datetime64[ns]"})
            self.df = self.df.rename(columns={"Feature (NaT)": TIME_COLUMN})

    def clean_humidity_data(self) -> None:
        """Clean humidity data by handling 'Bad' values."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        humidity_col = "大氣濕度 (R.H%)"
        if humidity_col in self.df.columns:
            # Replace "Bad" values with NaN
            self.df[humidity_col] = self.df[humidity_col].replace("Bad", np.nan)
            # Convert to numeric
            self.df[humidity_col] = pd.to_numeric(
                self.df[humidity_col], errors="coerce"
            )

    def fix_column_names(self) -> None:
        """Fix inconsistent column names."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Fix known column name issues
        column_fixes = {"冷卻水循環泵 G-511A 能耗 (KW)": "冷卻水循環泵G-511A 能耗 (KW)"}

        self.df = self.df.rename(columns=column_fixes)

    def calculate_cooling_system_cop(self) -> None:
        """Calculate Coefficient of Performance (COP) for cooling system based on thermodynamic principles."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Calculating cooling system COP...")

        # 1. Calculate heat rejection capacity of cooling tower
        # Q_rejection = ṁ_water × Cp_water × ΔT_water (kW)
        # Where: ṁ_water = mass flow rate (kg/s), Cp_water = 4.18 kJ/kg·K, ΔT = temp difference

        heat_rejection_capacity = 0

        # Method 1: If we have cooling water flow data and temperatures
        if all(col in self.df.columns for col in ["cooling_tower_outlet_temp_c", "cooling_tower_return_temp_c"]):
            temp_diff_tower = self.df["cooling_tower_return_temp_c"] - self.df["cooling_tower_outlet_temp_c"]

            # Estimate cooling water mass flow rate from pump power or use typical values
            # Typical cooling water flow: 2-4 L/s per kW of cooling capacity
            # For now, estimate based on system size or use available flow data
            estimated_flow_rate = 50  # kg/s (to be adjusted based on actual system)

            # Heat rejection (kW) = flow_rate (kg/s) × Cp (4.18 kJ/kg·K) × ΔT (K)
            heat_rejection_capacity = estimated_flow_rate * 4.18 * temp_diff_tower

            # Store heat rejection capacity
            self.df["cooling_tower_heat_rejection_kw"] = heat_rejection_capacity

        # Method 2: Estimate from individual cooler capacities
        cooler_heat_rejection = pd.Series(0.0, index=self.df.index)
        coolers = ["e401a", "e401b", "e401c", "e401d"]

        for cooler in coolers:
            flow_col = f"{cooler}_cooling_flow_m3_hr"
            outlet_temp_col = f"{cooler}_cooling_outlet_temp_c"

            if all(col in self.df.columns for col in [flow_col, outlet_temp_col]):
                # Convert m³/hr to kg/s (1 m³/hr ≈ 0.278 kg/s for water)
                flow_kg_s = self.df[flow_col] * 0.278

                # Assume inlet temperature or use cooling tower inlet temp
                if "cooling_tower_return_temp_c" in self.df.columns:
                    inlet_temp = self.df["cooling_tower_return_temp_c"]
                else:
                    inlet_temp = 35  # Typical inlet temperature

                temp_diff = inlet_temp - self.df[outlet_temp_col]
                cooler_capacity = flow_kg_s * 4.18 * temp_diff
                cooler_heat_rejection += cooler_capacity

        # Use the more accurate method
        if cooler_heat_rejection.sum() > 0:
            total_heat_rejection = cooler_heat_rejection
            self.df["total_heat_rejection_kw"] = total_heat_rejection
        else:
            total_heat_rejection = heat_rejection_capacity
            self.df["total_heat_rejection_kw"] = total_heat_rejection

        # 2. Calculate total cooling system power consumption
        fan_total_power = pd.Series(0.0, index=self.df.index)
        for fan_col in ["fan_510a_power_kw", "fan_510b_power_kw", "fan_510c_power_kw"]:
            if fan_col in self.df.columns:
                fan_total_power += self.df[fan_col]

        pump_total_power = pd.Series(0.0, index=self.df.index)
        for pump_col in ["cooling_pump_g511a_power_kw", "cooling_pump_g511b_power_kw",
                        "cooling_pump_g511c_power_kw", "cooling_pump_g511x_power_kw", "cooling_pump_g511y_power_kw"]:
            if pump_col in self.df.columns:
                pump_total_power += self.df[pump_col]

        # Store individual system powers
        self.df["fan_total_power_kw"] = fan_total_power
        self.df["pump_total_power_kw"] = pump_total_power

        # 3. Calculate COP for each subsystem

        # Fan system COP = Heat rejection by fans / Fan power
        # (Fans help with air-side heat transfer)
        if fan_total_power.sum() > 0:
            self.df["fan_system_cop"] = np.where(
                fan_total_power != 0,
                total_heat_rejection * 0.6 / fan_total_power,
                0
            )  # 0.6 factor represents fan contribution to total heat rejection
        else:
            self.df["fan_system_cop"] = 0

        # Pump system COP = Heat rejection by water circulation / Pump power
        if pump_total_power.sum() > 0:
            self.df["pump_system_cop"] = np.where(
                pump_total_power != 0,
                total_heat_rejection * 0.4 / pump_total_power,
                0
            )  # 0.4 factor represents pump contribution to total heat rejection
        else:
            self.df["pump_system_cop"] = 0

        # 4. Calculate overall cooling system COP
        total_cooling_power = fan_total_power + pump_total_power

        if total_cooling_power.sum() > 0:
            self.df["cooling_system_cop"] = np.where(
                total_cooling_power != 0,
                total_heat_rejection / total_cooling_power,
                0
            )

            # Cooling tower COP (same as overall system for this analysis)
            self.df["cooling_tower_cop"] = self.df["cooling_system_cop"]
        else:
            self.df["cooling_system_cop"] = 0
            self.df["cooling_tower_cop"] = 0

        # 5. Calculate weighted average cooling COP based on power consumption
        if total_cooling_power.sum() > 0:
            fan_weight = np.where(total_cooling_power != 0, fan_total_power / total_cooling_power, 0)
            pump_weight = np.where(total_cooling_power != 0, pump_total_power / total_cooling_power, 0)

            self.df["weighted_cooling_cop"] = np.where(
                total_cooling_power != 0,
                fan_weight * self.df["fan_system_cop"] + pump_weight * self.df["pump_system_cop"],
                0
            )
        else:
            self.df["weighted_cooling_cop"] = 0

        print("Cooling system COP calculation completed")

    def calculate_cooling_system_power(self) -> None:
        """Calculate cooling system total power consumption as target variable."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Calculate total cooling system power (fans + cooling tower)
        fan_power_sum = 0
        for fan_col in ["fan_510a_power_kw", "fan_510b_power_kw", "fan_510c_power_kw"]:
            if fan_col in self.df.columns:
                fan_power_sum += self.df[fan_col]

        # Calculate cooling pump power
        pump_power_sum = 0
        for pump_col in ["cooling_pump_g511a_power_kw", "cooling_pump_g511b_power_kw",
                        "cooling_pump_g511c_power_kw", "cooling_pump_g511x_power_kw", "cooling_pump_g511y_power_kw"]:
            if pump_col in self.df.columns:
                pump_power_sum += self.df[pump_col]

        # Total cooling system power
        self.df["cooling_tower_total_power_kw"] = fan_power_sum
        self.df["cooling_system_total_power_kw"] = fan_power_sum + pump_power_sum

    def calculate_cooling_load_rates(self) -> None:
        """Calculate cooling system load rates using 95th percentile method."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Calculating cooling system load rates using 95th percentile method...")

        # Calculate fan load rates
        fans = ["510a", "510b", "510c"]
        for fan in fans:
            power_col = f"fan_{fan}_power_kw"
            load_rate_col = f"fan_{fan}_load_rate"

            if power_col in self.df.columns:
                # Filter operational data (power > 0)
                operational_data = self.df[self.df[power_col] > 0][power_col]

                if len(operational_data) > 0:
                    # Calculate 95th percentile as reference maximum
                    ref_max_power = operational_data.quantile(LOAD_RATE_PERCENTILE / 100)

                    # Calculate load rate
                    self.df[load_rate_col] = self.df[power_col] / ref_max_power

                    print(f"Fan {fan.upper()}: Reference max power = {ref_max_power:.2f} kW")
                else:
                    print(f"Warning: No operational data found for fan {fan.upper()}")
                    self.df[load_rate_col] = 0

        # Calculate cooling tower load rate based on opening percentage
        if "cooling_tower_opening_pct" in self.df.columns:
            self.df["cooling_tower_load_rate"] = self.df["cooling_tower_opening_pct"] / 100

        # Calculate average cooling load rate
        cooling_load_rates = []
        for fan in fans:
            load_rate_col = f"fan_{fan}_load_rate"
            if load_rate_col in self.df.columns:
                cooling_load_rates.append(self.df[load_rate_col])

        if "cooling_tower_load_rate" in self.df.columns:
            cooling_load_rates.append(self.df["cooling_tower_load_rate"])

        if cooling_load_rates:
            avg_load_rate = sum(cooling_load_rates) / len(cooling_load_rates)
            self.df["avg_cooling_load_rate"] = avg_load_rate

        # Count operating cooling units
        operating_units = 0
        for fan in fans:
            power_col = f"fan_{fan}_power_kw"
            if power_col in self.df.columns:
                operating_units += (self.df[power_col] > OPERATION_THRESHOLD).astype(int)

        if "cooling_tower_opening_pct" in self.df.columns:
            operating_units += (self.df["cooling_tower_opening_pct"] > 5).astype(int)

        self.df["num_operating_cooling_units"] = operating_units

    def calculate_derived_features(self) -> None:
        """Calculate derived features for optimization model."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Calculating derived features...")

        # Calculate temperature differences
        machines = ["a", "b", "c"]
        for machine in machines:
            inlet_temp_col = f"rf501{machine}_chilled_inlet_temp_c"
            outlet_temp_col = f"rf501{machine}_chilled_outlet_temp_c"
            temp_diff_col = f"rf501{machine}_chilled_temp_diff"

            if all(col in self.df.columns for col in [inlet_temp_col, outlet_temp_col]):
                self.df[temp_diff_col] = (
                    self.df[inlet_temp_col] - self.df[outlet_temp_col]
                )

        # Calculate cooling capacity for each machine
        for machine in machines:
            flow_col = f"rf501{machine}_chilled_flow_m3_hr"
            temp_diff_col = f"rf501{machine}_chilled_temp_diff"
            cooling_capacity_col = f"rf501{machine}_cooling_capacity_kw"

            if all(col in self.df.columns for col in [flow_col, temp_diff_col]):
                # Cooling capacity = 1.163 × flow × temp_diff (kW)
                self.df[cooling_capacity_col] = (
                    1.163 * self.df[flow_col] * self.df[temp_diff_col]
                )

        print("Derived features calculated successfully")

    def calculate_multi_objective_targets(self) -> None:
        """Calculate multi-objective optimization targets for cooling system."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Calculating cooling system multi-objective targets...")

        # 1. Weighted cooling COP (already calculated)
        # Use cooling system COP as weighted average
        if "cooling_system_cop" in self.df.columns:
            self.df["weighted_cooling_cop"] = self.df["cooling_system_cop"]
        else:
            self.df["weighted_cooling_cop"] = 0

        # 2. Operational cost based on cooling system power
        if "cooling_system_total_power_kw" in self.df.columns:
            self.df["operational_cost"] = self.df["cooling_system_total_power_kw"]
        else:
            self.df["operational_cost"] = 0

        # 3. Load balance score for cooling system
        # Calculate load balance as inverse of coefficient of variation
        fans = ["510a", "510b", "510c"]
        load_rates = []

        for fan in fans:
            load_rate_col = f"fan_{fan}_load_rate"
            if load_rate_col in self.df.columns:
                load_rates.append(self.df[load_rate_col])

        # Add cooling tower load rate
        if "cooling_tower_load_rate" in self.df.columns:
            load_rates.append(self.df["cooling_tower_load_rate"])

        if load_rates:
            load_rates_array = np.array(load_rates).T

            # Calculate coefficient of variation for each row
            load_balance_scores = []
            for row in load_rates_array:
                operating_loads = row[row > 0.05]  # Only consider operating units (>5%)
                if len(operating_loads) > 1:
                    mean_load = np.mean(operating_loads)
                    std_load = np.std(operating_loads)
                    cv = std_load / mean_load if mean_load > 0 else 0
                    balance_score = 1 / (1 + cv)  # Higher score = better balance
                else:
                    balance_score = 1.0 if len(operating_loads) == 1 else 0.0
                load_balance_scores.append(balance_score)

            self.df["load_balance_score"] = load_balance_scores
        else:
            self.df["load_balance_score"] = 0

        print("Cooling system multi-objective targets calculated successfully")

    def handle_missing_values(self) -> None:
        """Handle missing values in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # For numerical columns, you might want to use different strategies
        # For now, we'll keep NaN values for feature engineering to handle
        print(f"Missing values summary:")
        missing_summary = self.df.isnull().sum()
        print(missing_summary[missing_summary > 0])

    def preprocess(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        print("Starting data preprocessing...")

        # Load data if not already loaded
        if self.df is None:
            self.load_data()

        # Apply all preprocessing steps
        self.clean_time_column()
        self.clean_humidity_data()
        self.clean_data_values()  # Clean data values first
        self.clean_column_names()  # Then clean column names
        self.fix_column_names()
        self.calculate_cooling_system_cop()
        self.calculate_cooling_system_power()
        self.calculate_cooling_load_rates()  # Calculate cooling system load rates
        self.calculate_derived_features()  # Calculate temperature differences and cooling capacities
        self.calculate_multi_objective_targets()  # Calculate optimization targets
        self.handle_missing_values()

        if self.df is not None:
            print(f"Preprocessing completed. Final shape: {self.df.shape}")
            return self.df
        else:
            raise ValueError("Preprocessing failed - DataFrame is None")

    def save_processed_data(self) -> None:
        """Save processed data to CSV file."""
        if self.df is None:
            raise ValueError("Data not processed. Call preprocess() first.")

        # Convert to Path if string
        output_path = (
            Path(self.output_path)
            if isinstance(self.output_path, str)
            else self.output_path
        )

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

    def run_preprocessing_pipeline(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline and save results."""
        processed_df = self.preprocess()
        self.save_processed_data()
        return processed_df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_preprocessing_pipeline()
    print("Data preprocessing completed successfully!")
