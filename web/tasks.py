import sys
import os
import json
import logging
from io import StringIO
import pandas as pd
from pathlib import Path
import requests

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Celery and Module Imports ---
from web.celery_app import celery_app
from constants import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    TRAINING_DATA_PATH, 
    CHILLER_BOUNDS,
    ALL_CONTROL_VARIABLES, 
    ALL_EXTERNAL_VARIABLES, 
    ALL_MONITORING_VARIABLES
)

logger = logging.getLogger(__name__)

# --- Celery Tasks ---

@celery_app.task(bind=True)
def process_data_task(self, csv_content: bytes, _filename: str):
    """Ingest raw CSV, run preprocessing + feature engineering, save unified processed dataset."""
    try:
        from src.data_processing import DataPreprocessor
        from src.feature_engineering import FeatureProcessor

        raw_df = pd.read_csv(StringIO(csv_content.decode('utf-8')))
        raw_df.to_csv(RAW_DATA_PATH, index=False)

        preprocessor = DataPreprocessor(input_path=Path(RAW_DATA_PATH), output_path=Path(PROCESSED_DATA_PATH))
        base_processed = preprocessor.preprocess()

        # Apply feature engineering and save as training data
        feature_processor = FeatureProcessor(verbose=False)
        fe_processed = feature_processor.process(base_processed)
        fe_processed.to_csv(TRAINING_DATA_PATH, index=False)

        logger.info(f"Data processing complete: {len(base_processed)} processed records saved to {PROCESSED_DATA_PATH}")
        logger.info(f"Feature engineering complete: {len(fe_processed)} training records saved to {TRAINING_DATA_PATH}")
        return {
            "status": "success",
            "records": len(fe_processed),
            "output": str(TRAINING_DATA_PATH),
            "features": len(fe_processed.columns) - 1  # Exclude time column
        }
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True)
def optimize_parameters_task(self, optimization_params: dict):
    """
    Run multi-objective optimization using the specified algorithm and parameters.
    This task is now connected to the real optimizer.
    """
    try:
        from src.optimizer import NSGA2Optimizer # Using direct class for now
        from constants import CHILLER_BOUNDS

        # Extract parameters from the input dictionary
        algorithm = optimization_params.get("algorithm", "nsga2")
        target_temp = optimization_params.get("target_temp", 7.0)
        population_size = optimization_params.get("population_size", 50)
        generations = optimization_params.get("max_iterations", 40)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        power_model_path = os.path.join(project_root, 'models', 'default_cooling_system_total_power_kw.pkl')
        cop_model_path = os.path.join(project_root, 'models', 'default_cooling_system_cop.pkl')

        # Load training data (with feature engineering applied)
        training_data_path = os.path.join(project_root, 'data', 'training_data.csv')
        try:
            # Load training data directly (already has feature engineering applied)
            training_df = pd.read_csv(training_data_path)
            
            # Get feature names (exclude target columns and time)
            exclude_columns = {'time', 'cooling_system_total_power_kw', 'cooling_system_cop'}
            actual_feature_names = [col for col in training_df.columns if col not in exclude_columns]
            
            # Calculate baseline values from training data (using median of last 100 records for recent typical values)
            recent_data = training_df.tail(100)  # Use recent data for more realistic baseline
            baseline_values = {}
            
            # For external variables (environment) and monitoring variables (current system state):
            # Use recent typical values from historical data
            for col in actual_feature_names:
                if col in recent_data.columns:
                    median_val = recent_data[col].median()
                    # Handle NaN values
                    if pd.isna(median_val):
                        mean_val = recent_data[col].mean()
                        baseline_values[col] = float(mean_val) if not pd.isna(mean_val) else 0.0
                    else:
                        baseline_values[col] = float(median_val)
                else:
                    baseline_values[col] = 0.0
            
            logger.info(f"Using {len(actual_feature_names)} features from training data")
            logger.info(f"Loaded baseline values for {len(baseline_values)} features from recent {len(recent_data)} records")
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}", exc_info=True)
            logger.warning(f"Trying to use processed data with feature engineering")
            
            # Fallback: Load processed data and apply feature engineering
            try:
                processed_data_path = os.path.join(project_root, 'data', 'processed.csv')
                processed_df = pd.read_csv(processed_data_path)
                
                from src.feature_engineering import FeatureProcessor
                processor = FeatureProcessor(verbose=False)
                training_df = processor.process(processed_df)
                
                exclude_columns = {'time', 'cooling_system_total_power_kw', 'cooling_system_cop'}
                actual_feature_names = [col for col in training_df.columns if col not in exclude_columns]
                
                recent_data = training_df.tail(100)
                baseline_values = {}
                for col in actual_feature_names:
                    if col in recent_data.columns:
                        median_val = recent_data[col].median()
                        baseline_values[col] = float(median_val) if not pd.isna(median_val) else 0.0
                    else:
                        baseline_values[col] = 0.0
                        
                logger.info(f"Generated features on-the-fly: {len(actual_feature_names)} features")
                
            except Exception as e2:
                logger.error(f"Failed to generate features on-the-fly: {str(e2)}", exc_info=True)
                logger.warning(f"Using fallback features")
                # Final fallback
                actual_feature_names = list(CHILLER_BOUNDS.keys()) + ['ambient_temperature_c', 'ambient_humidity_rh']
                baseline_values = {name: 0.0 for name in actual_feature_names}

        # Import variable classifications from constants
        # Create separate dictionaries for different variable types
        control_vars_from_features = [col for col in actual_feature_names if col in ALL_CONTROL_VARIABLES]
        external_vars_from_features = [col for col in actual_feature_names if col in ALL_EXTERNAL_VARIABLES]  
        monitoring_vars_from_features = [col for col in actual_feature_names if col in ALL_MONITORING_VARIABLES]
        other_vars = [col for col in actual_feature_names if col not in (ALL_CONTROL_VARIABLES + ALL_EXTERNAL_VARIABLES + ALL_MONITORING_VARIABLES)]
        
        logger.info(f"Variable classification:")
        logger.info(f"  - Control variables: {len(control_vars_from_features)} (will be optimized)")
        logger.info(f"  - External variables: {len(external_vars_from_features)} (fixed from environment)")
        logger.info(f"  - Monitoring variables: {len(monitoring_vars_from_features)} (fixed from current state)")
        logger.info(f"  - Other variables: {len(other_vars)} (fixed baseline)")

        # Create optimization bounds only for control variables that are in CHILLER_BOUNDS
        # and are also present in the feature set
        optimization_bounds = {}
        for var_name, bounds in CHILLER_BOUNDS.items():
            if var_name in control_vars_from_features:
                # Convert int bounds to float bounds for type compatibility
                optimization_bounds[var_name] = (float(bounds[0]), float(bounds[1]))
        
        logger.info(f"Optimization bounds for {len(optimization_bounds)} control variables: {list(optimization_bounds.keys())}")
        
        # NOTE: As discussed, we should use a factory. For this implementation, we use the class directly.
        # The factory pattern can be integrated here once all optimizer classes are fully implemented.
        optimizer = NSGA2Optimizer(
            bounds=optimization_bounds,
            power_model_path=power_model_path,
            cop_model_path=cop_model_path,
            feature_names=actual_feature_names
        )

        # Define fixed inputs for the optimization scenario using baseline values
        # This includes external variables (environment) and monitoring variables (current system state)
        fixed_inputs = baseline_values.copy()
        
        # Remove control variables from fixed inputs - these will be optimized
        for var_name in optimization_bounds.keys():
            if var_name in fixed_inputs:
                del fixed_inputs[var_name]
        
        # Get current system state to use as fixed environmental and monitoring conditions
        try:
            response = requests.get("http://api:8000/current-system-state", timeout=5)
            if response.status_code == 200:
                current_state = response.json()
                
                # Use current external variables (environment conditions)
                if "external_variables" in current_state:
                    for var_name, value in current_state["external_variables"].items():
                        if var_name in fixed_inputs:
                            fixed_inputs[var_name] = value
                            logger.info(f"Using current external variable {var_name}: {value}")
                
                # Use current monitoring variables (system state)
                if "monitoring_variables" in current_state:
                    for var_name, value in current_state["monitoring_variables"].items():
                        if var_name in fixed_inputs:
                            fixed_inputs[var_name] = value
                            logger.info(f"Using current monitoring variable {var_name}: {value}")
                
                logger.info("Successfully loaded current system state for optimization")
            else:
                logger.warning(f"Failed to get current system state: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not fetch current system state: {str(e)}, using baseline values")
        
        # Override with user-specified environmental values (if provided)
        ambient_temp = optimization_params.get('ambient_temp')
        if ambient_temp is not None:
            fixed_inputs['ambient_temperature_c'] = float(ambient_temp)
        ambient_humidity = optimization_params.get('ambient_humidity')
        if ambient_humidity is not None:
            fixed_inputs['ambient_humidity_rh'] = float(ambient_humidity)
        
        # Override with user-specified monitoring/system state values if provided
        for key, value in optimization_params.items():
            if key.startswith('current_') and key.replace('current_', '') in monitoring_vars_from_features:
                fixed_inputs[key.replace('current_', '')] = value
        
        logger.info(f"Fixed inputs include {len(fixed_inputs)} variables (environment + monitoring + other)")
        logger.info(f"Will optimize {len(optimization_bounds)} control variables: {list(optimization_bounds.keys())}")

        logger.info(f"Starting optimization with algorithm: {algorithm}...")
        optimization_results = optimizer.optimize(
            target_temp=target_temp,
            other_inputs=fixed_inputs,
            population_size=population_size,
            generations=generations
        )

        if not optimization_results or not optimization_results.get('solutions'):
            raise ValueError("Optimization did not return any valid solutions.")

        logger.info(f"Optimization completed successfully using {algorithm}.")
        return {"status": "success", "results": optimization_results}

    except Exception as e:
        logger.error(f"Optimization task failed: {str(e)}", exc_info=True)
        raise self.retry(countdown=60, max_retries=3, exc=e)

@celery_app.task(bind=True)
def train_model_task(self, model_name: str, target_column: str):
    """
    Train a prediction model using the specified model name and target column.
    """
    try:
        from src.prediction_models import train_cooling_system_model
        
        logger.info(f"Starting training for {target_column} using {model_name} model...")
        
        model, metrics = train_cooling_system_model(
            model_name=model_name,
            target_columns=[target_column]
        )
        
        logger.info(f"Model training for {target_column} complete. Metrics: {metrics}")
        
        return {
            "status": "success",
            "model_name": model.model_name,
            "target": target_column,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Model training failed for {target_column}: {str(e)}", exc_info=True)
        raise self.retry(countdown=60, max_retries=3, exc=e)


# Export the celery app and tasks
__all__ = ['celery_app', 'process_data_task', 'train_model_task', 'optimize_parameters_task']
