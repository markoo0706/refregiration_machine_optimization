"""
冷卻塔系統特徵工程模組 - 包含時間序列滯後特徵
========================================================

重點功能：
1. 時間序列滯後特徵（-4 lagging features）
2. 冷卻塔與風扇系統特徵工程
3. 控制變數與監控變數處理
4. 目標預測特徵準備

設計原則：
- 專注於冷卻塔和風扇系統
- 時間序列預測導向
- 使用過去一小時（4個15分鐘時間點）的數據
- 支援多種預測目標（溫度、功率等）

"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 導入冷卻塔相關常數
from web.constants import (
    DATA_DIR,
    UNIFIED_FEATURES,
    ALL_CONTROL_VARIABLES,
    ALL_MONITORING_VARIABLES,
    ALL_EXTERNAL_VARIABLES,
    PRIMARY_ENERGY_TARGETS,
    EFFICIENCY_TARGETS,
    BALANCE_TARGETS,
    COOLING_TOWER_CONTROL_VARIABLES,
    COOLER_CONTROL_VARIABLES,
    FAN_CONTROL_VARIABLES,
    PUMP_CONTROL_VARIABLES,
    TEMPERATURE_MONITORING_VARIABLES,
    FLOW_MONITORING_VARIABLES,
    CURRENT_MONITORING_VARIABLES,
    WEATHER_VARIABLES,
    PROCESS_DEMAND_VARIABLES,
    TIME_COLUMN,
)

# 抑制 pandas 警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class FeatureProcessor:
    """冷卻塔系統特徵處理器 - 專注於時間序列特徵工程

    功能:
    - 時間序列滯後特徵（-1, -2, -3, -4 lag）
    - 控制變數與監控變數特徵工程
    - 滾動窗口統計特徵
    - 預測目標準備
    """

    def __init__(self,
                 lag_periods: int = 4,
                 time_column: str = TIME_COLUMN,
                 verbose: bool = True):
        """
        參數:
        - lag_periods: 滯後期數，預設4（過去1小時，15分鐘/筆）
        - time_column: 時間欄位名稱
        - verbose: 是否顯示詳細資訊
        """
        self.lag_periods = lag_periods
        self.time_column = time_column
        self.verbose = verbose

        # 特徵分類
        self.control_features = []
        self.monitoring_features = []
        self.lagging_features = []
        self.target_features = []
        self.all_features = []

        # 模型相關
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance_cache = {}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """主要處理流程：完整特徵工程管道

        參數:
        - df: 原始數據

        回傳:
        - 處理後包含所有特徵的DataFrame
        """
        if self.verbose:
            print("=== 冷卻塔系統特徵工程 ===")
            print(f"原始數據形狀: {df.shape}")

        # 1. 數據清理與預處理
        df_clean = self._clean_data(df.copy())

        # 2. 基礎特徵工程
        df_features = self._create_base_features(df_clean)

        # 3. 創建滯後特徵
        df_lagged = self._create_lagging_features(df_features)

        # 4. 滾動窗口統計特徵
        df_rolling = self._create_rolling_features(df_lagged)

        # 5. 目標變數工程
        df_final = self._create_target_features(df_rolling)

        # 6. 移除包含NaN的行（由於滯後特徵產生）
        df_final = df_final.dropna().reset_index(drop=True)

        if self.verbose:
            print(f"最終特徵數據形狀: {df_final.shape}")
            print(f"移除了前{self.lag_periods}行（滯後特徵NaN）")
            self._print_feature_summary()

        return df_final

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """數據清理"""
        if self.verbose:
            print("執行數據清理...")

        # 時間處理
        if self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
            df = df.sort_values(self.time_column).reset_index(drop=True)

        # 功率數據清理（不能為負數且合理範圍）
        power_columns = [col for col in FAN_CONTROL_VARIABLES + PUMP_CONTROL_VARIABLES if col in df.columns]
        for col in power_columns:
            df[col] = df[col].clip(lower=0, upper=1000)  # 0-1000 kW

        # 溫度數據清理
        temp_columns = [col for col in TEMPERATURE_MONITORING_VARIABLES if col in df.columns]
        temp_columns += [col for col in WEATHER_VARIABLES if 'temp' in col.lower()]
        for col in temp_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=-10, upper=60)  # -10~60°C

        # 百分比數據清理（開度等）
        pct_columns = [col for col in df.columns if 'pct' in col or 'opening' in col]
        for col in pct_columns:
            df[col] = df[col].clip(lower=0, upper=100)  # 0-100%

        # 處理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0)

        return df

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建基礎特徵"""
        if self.verbose:
            print("創建基礎特徵...")

        features = df.copy()

        # 1. 控制變數特徵
        self._add_control_features(features)

        # 2. 監控變數特徵
        self._add_monitoring_features(features)

        # 3. 衍生特徵
        self._add_derived_features(features)

        return features

    def _add_control_features(self, df: pd.DataFrame):
        """新增控制變數特徵"""

        # 冷卻塔控制變數
        for col in COOLING_TOWER_CONTROL_VARIABLES:
            if col in df.columns:
                self.control_features.append(col)

                # 非線性特徵
                if 'opening' in col or 'pct' in col:
                    # 開度平方特徵（非線性關係）
                    df[f"{col}_squared"] = (df[col] / 100.0) ** 2
                    self.control_features.append(f"{col}_squared")

        # 冷卻器控制變數
        for col in COOLER_CONTROL_VARIABLES:
            if col in df.columns:
                self.control_features.append(col)

        # 風扇控制變數
        for col in FAN_CONTROL_VARIABLES:
            if col in df.columns:
                self.control_features.append(col)

                # 功率特徵工程
                if 'power' in col:
                    # 功率平方根（風扇特性）
                    df[f"{col}_sqrt"] = np.sqrt(np.maximum(df[col], 0))
                    self.control_features.append(f"{col}_sqrt")

        # 泵浦控制變數
        for col in PUMP_CONTROL_VARIABLES:
            if col in df.columns:
                self.control_features.append(col)

    def _add_monitoring_features(self, df: pd.DataFrame):
        """新增監控變數特徵"""

        # 溫度監控
        for col in TEMPERATURE_MONITORING_VARIABLES:
            if col in df.columns:
                self.monitoring_features.append(col)

        # 流量監控
        for col in FLOW_MONITORING_VARIABLES:
            if col in df.columns:
                self.monitoring_features.append(col)

        # 電流監控
        for col in CURRENT_MONITORING_VARIABLES:
            if col in df.columns:
                self.monitoring_features.append(col)

        # 外部環境變數
        for col in ALL_EXTERNAL_VARIABLES:
            if col in df.columns:
                self.monitoring_features.append(col)

    def _add_derived_features(self, df: pd.DataFrame):
        """新增衍生特徵"""

        # 1. 系統總計特徵
        fan_power_cols = [col for col in FAN_CONTROL_VARIABLES if col in df.columns and 'power' in col]
        if fan_power_cols:
            df['total_fan_power_kw'] = df[fan_power_cols].sum(axis=1)
            self.control_features.append('total_fan_power_kw')

        pump_power_cols = [col for col in PUMP_CONTROL_VARIABLES if col in df.columns]
        if pump_power_cols:
            df['total_pump_power_kw'] = df[pump_power_cols].sum(axis=1)
            self.control_features.append('total_pump_power_kw')

        # 2. 溫差特徵
        if 'cooling_tower_outlet_temp_c' in df.columns and 'cooling_tower_return_temp_c' in df.columns:
            df['cooling_tower_temp_diff'] = df['cooling_tower_return_temp_c'] - df['cooling_tower_outlet_temp_c']
            self.monitoring_features.append('cooling_tower_temp_diff')

        # 3. 環境負載交互作用
        if 'ambient_temperature_c' in df.columns and 'total_fan_power_kw' in df.columns:
            df['ambient_fan_interaction'] = df['ambient_temperature_c'] * df['total_fan_power_kw']
            self.monitoring_features.append('ambient_fan_interaction')

        # 4. 效率指標
        if 'total_fan_power_kw' in df.columns and df['total_fan_power_kw'].sum() > 0:
            # 運行風扇數量
            df['active_fans_count'] = (df[fan_power_cols] > 5).sum(axis=1)  # 功率>5kW視為運行
            self.monitoring_features.append('active_fans_count')

    def _create_lagging_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建滯後特徵 - 核心功能"""
        if self.verbose:
            print(f"創建 -{self.lag_periods} 滯後特徵...")

        # 選擇要創建滯後特徵的變數
        lag_variables = []

        # 重要控制變數
        important_controls = [
            'cooling_tower_opening_pct',
            'total_fan_power_kw',
            'total_pump_power_kw'
        ]
        lag_variables.extend([col for col in important_controls if col in df.columns])

        # 關鍵監控變數
        important_monitoring = [
            'ambient_temperature_c',
            'cooling_tower_outlet_temp_c',
            'cooling_tower_return_temp_c',
            'cooling_tower_temp_diff',
            'ambient_humidity_rh'
        ]
        lag_variables.extend([col for col in important_monitoring if col in df.columns])

        # 電流監控（系統負載指標）
        current_cols = [col for col in CURRENT_MONITORING_VARIABLES if col in df.columns]
        lag_variables.extend(current_cols[:3])  # 只取前3個避免特徵過多

        if self.verbose:
            print(f"對 {len(lag_variables)} 個變數創建滯後特徵")

        df_lagged = df.copy()

        # 為每個變數創建 -1 到 -lag_periods 的滯後特徵
        for variable in lag_variables:
            if variable not in df.columns:
                continue

            for lag in range(1, self.lag_periods + 1):
                lag_col_name = f"{variable}_lag{lag}"
                df_lagged[lag_col_name] = df[variable].shift(lag)
                self.lagging_features.append(lag_col_name)

        if self.verbose:
            print(f"創建了 {len(self.lagging_features)} 個滯後特徵")

        return df_lagged

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建滾動窗口統計特徵"""
        if self.verbose:
            print("創建滾動窗口統計特徵...")

        # 關鍵變數的滾動統計
        key_variables = [
            'total_fan_power_kw',
            'ambient_temperature_c',
            'cooling_tower_temp_diff'
        ]

        rolling_features = []

        for variable in key_variables:
            if variable not in df.columns:
                continue

            # 4期（1小時）滾動平均
            df[f"{variable}_rolling_mean"] = df[variable].rolling(window=self.lag_periods, min_periods=1).mean()
            rolling_features.append(f"{variable}_rolling_mean")

            # 4期滾動標準差（波動性）
            df[f"{variable}_rolling_std"] = df[variable].rolling(window=self.lag_periods, min_periods=1).std().fillna(0)
            rolling_features.append(f"{variable}_rolling_std")

        self.monitoring_features.extend(rolling_features)

        if self.verbose:
            print(f"創建了 {len(rolling_features)} 個滾動統計特徵")

        return df

    def _create_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建目標變數特徵"""
        if self.verbose:
            print("創建目標變數特徵...")

        # 主要能耗目標
        for target in PRIMARY_ENERGY_TARGETS:
            if target in df.columns:
                self.target_features.append(target)
            elif target == 'cooling_system_total_power_kw':
                # 計算總冷卻系統功率
                fan_power = df.get('total_fan_power_kw', 0)
                pump_power = df.get('total_pump_power_kw', 0)
                df[target] = fan_power + pump_power
                self.target_features.append(target)

        # 效率目標
        for target in EFFICIENCY_TARGETS:
            if target in df.columns:
                self.target_features.append(target)
            elif target == 'cooling_system_cop':
                # 簡化COP計算
                total_power = df.get('cooling_system_total_power_kw', 1)
                cooling_capacity = df.get('cooling_tower_temp_diff', 5) * 100  # 簡化冷卻能力估算
                df[target] = np.where(total_power > 0, cooling_capacity / total_power, 0)
                self.target_features.append(target)

        return df

    def get_training_data(self, df: pd.DataFrame,
                         target_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """準備訓練數據

        參數:
        - df: 特徵工程後的數據
        - target_columns: 目標欄位列表，None則使用所有目標

        回傳:
        - (X, y): 特徵矩陣和目標矩陣
        """

        # 使用統一的特徵集
        self.all_features = UNIFIED_FEATURES

        available_features = [col for col in self.all_features if col in df.columns]

        if target_columns is None:
            target_columns = [col for col in self.target_features if col in df.columns]
        else:
            target_columns = [col for col in target_columns if col in df.columns]

        if self.verbose:
            print(f"可用特徵: {len(available_features)} 個")
            print(f"目標變數: {len(target_columns)} 個")

        # 構建特徵和目標矩陣
        feature_cols = [self.time_column] + available_features if self.time_column in df.columns else available_features
        target_cols = [self.time_column] + target_columns if self.time_column in df.columns else target_columns

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        # 移除包含NaN的行
        if available_features and target_columns:
            valid_mask = ~(X[available_features].isna().any(axis=1) |
                          y[target_columns].isna().any(axis=1))
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)

        return X, y

    def _print_feature_summary(self):
        """打印特徵摘要"""
        if not self.verbose:
            return

        print("\n=== 特徵工程摘要 ===")
        print(f"控制變數特徵: {len(self.control_features)} 個")
        print(f"監控變數特徵: {len(self.monitoring_features)} 個")
        print(f"滯後特徵: {len(self.lagging_features)} 個")
        print(f"目標變數: {len(self.target_features)} 個")
        print(f"總特徵數: {len(self.all_features)} 個")

        if self.lagging_features:
            print(f"\n滯後特徵範例:")
            for feature in self.lagging_features[:5]:  # 只顯示前5個
                print(f"  - {feature}")
            if len(self.lagging_features) > 5:
                print(f"  ... 還有 {len(self.lagging_features) - 5} 個")


# -------------------------------------------------------------------------------------
# 向後兼容的簡化介面
# -------------------------------------------------------------------------------------

def train(input_path: Optional[str] = None,
          output_path: Optional[str] = None,
          verbose: bool = True) -> Tuple[FeatureProcessor, pd.DataFrame, pd.DataFrame]:
    """執行特徵工程並準備訓練數據

    回傳: (processor, X, y)
    """
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "processed.csv")
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "training_data.csv")

    # 讀取數據
    df = pd.read_csv(input_path)

    # 創建特徵處理器並處理數據
    processor = FeatureProcessor(verbose=verbose)
    df_processed = processor.process(df)

    # 準備訓練數據
    X, y = processor.get_training_data(df_processed)

    # 保存處理後的數據
    if output_path:
        combined = pd.merge(X, y, on=TIME_COLUMN, how='inner') if TIME_COLUMN in X.columns else pd.concat([X, y], axis=1)
        combined.to_csv(output_path, index=False)
        if verbose:
            print(f"訓練數據已保存至: {output_path}")

    return processor, X, y


def predict(processor: FeatureProcessor,
           new_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """使用已訓練的處理器進行特徵工程

    參數:
    - processor: 已訓練的FeatureProcessor
    - new_data: 新數據（路徑或DataFrame）

    回傳:
    - 特徵工程後的DataFrame
    """
    if isinstance(new_data, str):
        df = pd.read_csv(new_data)
    else:
        df = new_data.copy()

    return processor.process(df)


__all__ = [
    "FeatureProcessor",
    "train",
    "predict"
]