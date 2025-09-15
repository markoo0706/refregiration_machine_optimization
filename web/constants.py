"""冰水主機最佳化系統 - 特徵與變數分類常數定義

此檔案定義了用於冰水主機能耗預測模型和最佳化算法的所有變數分類。
按功能和控制性質將變數分為以下幾大類：
- 控制變數：可以主動調控的設備參數
- 外部環境變數：無法控制的環境因子
- 能耗目標變數：用於最佳化的目標函數
- 輔助與計算變數：從原始數據計算得出的特徵
"""

from __future__ import annotations
import os

# ======================================================================================
# 專案路徑配置
# ======================================================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# 統一使用單一檔案路徑（避免原本指向資料夾導致讀檔失敗）
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
SRC_DIR = os.path.join(PROJECT_ROOT, "app", "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 模型訓練參數
TEST_SIZE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

# 模型檔案路徑（簡化示例，可再引入時間戳機制）
POWER_MODEL_PATH = os.path.join(MODELS_DIR, "power_model.pkl")
TEMPERATURE_MODEL_PATH = os.path.join(MODELS_DIR, "temperature_model.pkl")
COP_MODEL_PATH = os.path.join(MODELS_DIR, "cop_model.pkl")




# ======================================================================================
# 控制變數 (Decision Variables) - 可主動調控的操作參數
# ======================================================================================

# 閥門開度控制變數
VALVE_CONTROL_VARIABLES = [
    "tic607a_cooling_valve_opening_pct",  # 反應釜A冷卻水閥
    "tic607b_cooling_valve_opening_pct",  # 反應釜B冷卻水閥
    "tic607c_cooling_valve_opening_pct",  # 反應釜C冷卻水閥
    "tic607d_cooling_valve_opening_pct",  # 反應釜D冷卻水閥
    "cooling_tower_opening_pct",  # 冷卻水塔閥門開度
]

# 冷卻水塔系統控制變數
COOLING_TOWER_CONTROL_VARIABLES = [
    "cooling_tower_opening_pct",  # 冷卻水塔開度
    "cooling_tower_outlet_temp_c",  # 冷卻水塔出水溫度設定值
    "cooling_tower_return_temp_c",  # 冷卻水塔回水溫度設定值
]

# 冷卻器系統控制變數
COOLER_CONTROL_VARIABLES = [
    "e401a_cooling_outlet_temp_c",  # 冷卻器A出口溫度
    "e401b_cooling_outlet_temp_c",  # 冷卻器B出口溫度
    "e401c_cooling_outlet_temp_c",  # 冷卻器C出口溫度
    "e401d_cooling_outlet_temp_c",  # 冷卻器D出口溫度
    "e401a_cooling_flow_m3_hr",     # 冷卻器A流量
    "e401b_cooling_flow_m3_hr",     # 冷卻器B流量
    "e401c_cooling_flow_m3_hr",     # 冷卻器C流量
    "e401d_cooling_flow_m3_hr",     # 冷卻器D流量
    "cooling_water_outlet_pressure_kg_cm2",  # 冷卻水出水壓力
]

# 泵浦系統控制變數
PUMP_CONTROL_VARIABLES = [
    "cooling_pump_g511a_power_kw",  # 冷卻水循環泵
    "cooling_pump_g511b_power_kw",
    "cooling_pump_g511c_power_kw",
    "cooling_pump_g511x_power_kw",
    "cooling_pump_g511y_power_kw",
    "chilled_pump_g501a_power_kw",  # 冷凍水循環泵
    "chilled_pump_g501b_power_kw",
    "chilled_pump_g501c_power_kw",
]

# 風扇系統控制變數
FAN_CONTROL_VARIABLES = [
    "fan_510a_power_kw",
    "fan_510b_power_kw",
    "fan_510c_power_kw",
]

# 集成所有控制變數
ALL_CONTROL_VARIABLES = (
    VALVE_CONTROL_VARIABLES
    + COOLING_TOWER_CONTROL_VARIABLES
    + COOLER_CONTROL_VARIABLES
    + PUMP_CONTROL_VARIABLES
    + FAN_CONTROL_VARIABLES
)

# ======================================================================================
# 外部環境變數 (External Variables) - 無法控制的環境因子
# ======================================================================================

# 氣象環境變數
WEATHER_VARIABLES = [
    "ambient_temperature_c",  # 大氣溫度
    "ambient_humidity_rh",  # 大氣濕度
]

# 生產製程需求變數（來自反應釜等製程設備）
PROCESS_DEMAND_VARIABLES = [
    "tic206a_reactor_temp_c",  # 反應釜A溫度需求
    "tic206b_reactor_temp_c",  # 反應釜B溫度需求
    "tic206c_reactor_temp_c",  # 反應釜C溫度需求
    "tic206d_reactor_temp_c",  # 反應釜D溫度需求
    "tic607a_reactor_oil_temp_c",  # 反應釜A油溫
    "tic607b_reactor_oil_temp_c",  # 反應釜B油溫
    "tic607c_reactor_oil_temp_c",  # 反應釜C油溫
    "tic607d_reactor_oil_temp_c",  # 反應釜D油溫
]

# 集成所有外部環境變數
ALL_EXTERNAL_VARIABLES = WEATHER_VARIABLES + PROCESS_DEMAND_VARIABLES

# ======================================================================================
# 能耗目標變數 (Energy Consumption Targets) - 最佳化目標
# ======================================================================================

# 主要能耗變數（最佳化主要目標）- 重點關注冷卻水塔和風扇系統
PRIMARY_ENERGY_TARGETS = [
    "fan_510a_power_kw",  # 風扇A能耗
    "fan_510b_power_kw",  # 風扇B能耗
    "fan_510c_power_kw",  # 風扇C能耗
    "cooling_tower_total_power_kw",  # 冷卻水塔總能耗
    "cooling_system_total_power_kw",  # 冷卻系統總能耗
]


EFFICIENCY_TARGETS = [
    "cooling_system_cop",    # 整個冷卻水系統COP
    "fan_system_cop",        # 風扇系統COP
    "cooling_tower_cop",     # 冷卻水塔COP
    "weighted_cooling_cop",  # 加權平均冷卻系統COP
]

# 平衡指標目標變數
BALANCE_TARGETS = [
    "load_balance_score",  # 負載平衡分數
]

# 集成所有目標變數
ALL_TARGET_VARIABLES = PRIMARY_ENERGY_TARGETS + EFFICIENCY_TARGETS + BALANCE_TARGETS

# ======================================================================================
# 狀態監控變數 (State Monitoring) - 系統狀態指標
# ======================================================================================

# 溫度監控變數（移除已變為控制變數的項目）
TEMPERATURE_MONITORING_VARIABLES = [
    "rf501a_chilled_inlet_temp_c",  # 冷凍機A冷凍水入口溫度
    "rf501a_chilled_outlet_temp_c",  # 冷凍機A冷凍水出口溫度
    "rf501a_cooling_inlet_temp_c",  # 冷凍機A冷卻水入口溫度
    "rf501a_cooling_outlet_temp_c",  # 冷凍機A冷卻水出口溫度
    "rf501b_chilled_inlet_temp_c",  # 冷凍機B冷凍水入口溫度
    "rf501b_chilled_outlet_temp_c",  # 冷凍機B冷凍水出口溫度
    "rf501b_cooling_inlet_temp_c",  # 冷凍機B冷卻水入口溫度
    "rf501b_cooling_outlet_temp_c",  # 冷凍機B冷卻水出口溫度
    "rf501c_chilled_inlet_temp_c",  # 冷凍機C冷凍水入口溫度
    "rf501c_chilled_outlet_temp_c",  # 冷凍機C冷凍水出口溫度
    "rf501c_cooling_inlet_temp_c",  # 冷凍機C冷卻水入口溫度
    "rf501c_cooling_outlet_temp_c",  # 冷凍機C冷卻水出口溫度
    "chilled_water_return_temp_c",  # 冷凍水回流溫度
    "chilled_water_tank_temp_c",  # 冷凍水槽溫度
]

# 流量監控變數（移除已變為控制變數的項目）
FLOW_MONITORING_VARIABLES = [
    "rf501a_chilled_flow_m3_hr",  # 冷凍機A冷凍水流量
    "rf501b_chilled_flow_m3_hr",  # 冷凍機B冷凍水流量
    "rf501c_chilled_flow_m3_hr",  # 冷凍機C冷凍水流量
]

# 壓力監控變數（移除已變為控制變數的項目）
PRESSURE_MONITORING_VARIABLES = [
    "chilled_water_pressure_diff_kg_cm2",  # 冷凍水壓差
]

# 電流監控變數
CURRENT_MONITORING_VARIABLES = [
    "rf501a_current_a",  # 冷凍機A電流
    "rf501b_current_a",  # 冷凍機B電流
    "rf501c_current_a",  # 冷凍機C電流
    "fan_510a_current_a",  # 風扇A電流
    "fan_510b_current_a",  # 風扇B電流
    "fan_510c_current_a",  # 風扇C電流
]

# 集成所有監控變數
ALL_MONITORING_VARIABLES = (
    TEMPERATURE_MONITORING_VARIABLES
    + FLOW_MONITORING_VARIABLES
    + PRESSURE_MONITORING_VARIABLES
    + CURRENT_MONITORING_VARIABLES
)

# ======================================================================================
# 統一模型特徵 (Unified Model Features)
# ======================================================================================

UNIFIED_FEATURES = list(set(
    ALL_CONTROL_VARIABLES +
    ALL_MONITORING_VARIABLES +
    ALL_EXTERNAL_VARIABLES
))


# ======================================================================================
# 計算與衍生特徵變數 (Derived Features)
# ======================================================================================

# 溫差計算特徵
TEMPERATURE_DIFF_FEATURES = [
    "rf501a_chilled_temp_diff",  # 冷凍機A冷凍水溫差
    "rf501b_chilled_temp_diff",  # 冷凍機B冷凍水溫差
    "rf501c_chilled_temp_diff",  # 冷凍機C冷凍水溫差
]


# 負載與需求特徵（改為冷卻機組：風扇+冷卻水塔）
LOAD_DEMAND_FEATURES = [
    "fan_510a_load_rate",         # 風扇A負載率
    "fan_510a_load_rate_squared", # 風扇A負載率平方
    "fan_510a_load_rate_cubed",   # 風扇A負載率立方
    "fan_510b_load_rate",         # 風扇B負載率
    "fan_510b_load_rate_squared", # 風扇B負載率平方
    "fan_510b_load_rate_cubed",   # 風扇B負載率立方
    "fan_510c_load_rate",         # 風扇C負載率
    "fan_510c_load_rate_squared", # 風扇C負載率平方
    "fan_510c_load_rate_cubed",   # 風扇C負載率立方
    "cooling_tower_load_rate",    # 冷卻水塔負載率
    "cooling_tower_load_rate_squared", # 冷卻水塔負載率平方
    "avg_cooling_load_rate",      # 平均冷卻負載率
    "num_operating_cooling_units", # 運行冷卻機組數量
]


# 集成所有衍生特徵
ALL_DERIVED_FEATURES = TEMPERATURE_DIFF_FEATURES + LOAD_DEMAND_FEATURES

# ======================================================================================
# 溫度模型特徵 (Temperature Model Features)
# ======================================================================================
# TemperaturePredictModel 主要使用的核心特徵集合，避免直接包含總能耗與個別機組功率，
# 以免資訊洩漏；可再由訓練流程做特徵選擇。後續可擴充加入滯後特徵、移動平均等。
TEMPERATURE_MODEL_BASE_FEATURES = (
    TEMPERATURE_MONITORING_VARIABLES
    + FLOW_MONITORING_VARIABLES
    + WEATHER_VARIABLES
    + FAN_CONTROL_VARIABLES
    + VALVE_CONTROL_VARIABLES
)

# (原本動態 append 已移除，統一於文件底部 __all__ 宣告匯出)

# ======================================================================================
# 最佳化相關參數與限制
# ======================================================================================

# 負載率參數
LOAD_RATE_PERCENTILE = 95
OPERATION_THRESHOLD = 0.05
MAX_LOAD_RATE = 1.2
MIN_LOAD_RATE = 0.1

# 最佳化目標權重
OPTIMIZATION_TARGETS = {
    "total_power": "total_power_kw",  # 最小化總能耗
    "weighted_cop": "weighted_avg_cop",  # 最大化效率
    "operational_cost": "operational_cost",  # 最小化營運成本
    "load_balance_score": "load_balance_score",  # 最佳化負載平衡
}

# 參數搜尋範圍（示意用：實際應依設備規格調整）
CHILLER_BOUNDS = {
    "cooling_tower_opening_pct": (0, 100),
    "fan_510a_power_kw": (0, 160),
    "fan_510b_power_kw": (0, 160),
    "fan_510c_power_kw": (0, 160),
    # 可擴充其他決策變數（泵、閥門開度等）
}

# 決策變數定義（用於最佳化算法）
DECISION_VARIABLES = {
    "target_load_rates": [
        "rf501a_target_load_rate",
        "rf501b_target_load_rate",
        "rf501c_target_load_rate",
    ],
    "valve_openings": [
        "tic607a_cooling_valve_target_opening",
        "tic607b_cooling_valve_target_opening",
        "tic607c_cooling_valve_target_opening",
        "tic607d_cooling_valve_target_opening",
        "cooling_tower_target_opening",
    ],
}

# 約束條件參數
OPERATIONAL_CONSTRAINTS = {
    "min_operating_machines": 1,  # 最少運行機組數
    "max_operating_machines": 3,  # 最多運行機組數
    "min_total_capacity": 100,  # 最小總冷卻能力 (kW)
    "max_total_power": 600,  # 最大總能耗 (kW)
    "min_system_cop": 2.5,  # 最小系統COP
    "max_load_imbalance": 0.3,  # 最大負載不平衡度
}

# 時間參數
TIME_COLUMN = "time"


# ======================================================================================
# 欄位名稱對應表 (Raw Data to Processed Data)
# ======================================================================================

COLUMN_MAPPING = {
    "Time": "time",
    "FAN-510A能耗 (KW)": "fan_510a_power_kw",
    "FAN-510B能耗 (KW)": "fan_510b_power_kw",
    "FAN-510C能耗 (KW)": "fan_510c_power_kw",
    "FAN-510A電流值 (A)": "fan_510a_current_a",
    "FAN-510B電流值 (A)": "fan_510b_current_a",
    "FAN-510C電流值 (A)": "fan_510c_current_a",
    "冷卻水塔出水溫度 (℃)": "cooling_tower_outlet_temp_c",
    "冷卻水塔回水溫度 (℃)": "cooling_tower_return_temp_c",
    "冷卻水出水壓力 (KG/cm2)": "cooling_water_outlet_pressure_kg_cm2",
    "大氣溫度 (℃)": "ambient_temperature_c",
    "大氣濕度 (R.H%)": "ambient_humidity_rh",
    "冷卻水循環泵G-511A 能耗 (KW)": "cooling_pump_g511a_power_kw",
    "冷卻水循環泵G-511B 能耗 (KW)": "cooling_pump_g511b_power_kw",
    "冷卻水循環泵G-511C 能耗 (KW)": "cooling_pump_g511c_power_kw",
    "冷卻水循環泵G-511X 能耗 (KW)": "cooling_pump_g511x_power_kw",
    "冷卻水循環泵G-511Y 能耗 (KW)": "cooling_pump_g511y_power_kw",
    "冷凍機RF-501A能耗 (KW)": "chiller_rf501a_power_kw",
    "冷凍機RF-501B能耗 (KW)": "chiller_rf501b_power_kw",
    "冷凍機RF-501C能耗 (KW)": "chiller_rf501c_power_kw",
    "冷凍水回流溫度 (℃)": "chilled_water_return_temp_c",
    "冷凍水槽溫度 (℃)": "chilled_water_tank_temp_c",
    "冷凍水循環泵G-501A能耗 (KW)": "chilled_pump_g501a_power_kw",
    "冷凍水循環泵G-501B能耗 (KW)": "chilled_pump_g501b_power_kw",
    "冷凍水循環泵G-501C能耗 (KW)": "chilled_pump_g501c_power_kw",
    "RF-501A 冷凍水入口溫度 (℃)": "rf501a_chilled_inlet_temp_c",
    "RF-501A 冷凍水出口溫度 (℃)": "rf501a_chilled_outlet_temp_c",
    "RF-501A 冷凍水流量 (l/min)": "rf501a_chilled_flow_l_min",
    "RF-501A 冷卻水入口溫度 (℃)": "rf501a_cooling_inlet_temp_c",
    "RF-501A 冷卻水出口溫度 (℃)": "rf501a_cooling_outlet_temp_c",
    "RF-501B 冷凍水入口溫度 (℃)": "rf501b_chilled_inlet_temp_c",
    "RF-501B 冷凍水出口溫度 (℃)": "rf501b_chilled_outlet_temp_c",
    "RF-501B 冷凍水流量 (l/min)": "rf501b_chilled_flow_l_min",
    "RF-501B 冷卻水入口溫度 (℃)": "rf501b_cooling_inlet_temp_c",
    "RF-501B 冷卻水出口溫度 (℃)": "rf501b_cooling_outlet_temp_c",
    "RF-501C 冷凍水入口溫度 (℃)": "rf501c_chilled_inlet_temp_c",
    "RF-501C 冷凍水出口溫度 (℃)": "rf501c_chilled_outlet_temp_c",
    "RF-501C 冷凍水流量 (l/min)": "rf501c_chilled_flow_l_min",
    "RF-501C 冷卻水入口溫度 (℃)": "rf501c_cooling_inlet_temp_c",
    "RF-501C 冷卻水出口溫度 (℃)": "rf501c_cooling_outlet_temp_c",
    "TIC-206A\n反應釜溫度 (℃)": "tic206a_reactor_temp_c",
    "TIC-607A\n反應釜油溫 (℃)": "tic607a_reactor_oil_temp_c",
    "TIC-607A\n冷卻水閥開度 (%)": "tic607a_cooling_valve_opening_pct",
    "TIC-206B\n反應釜溫度 (℃)": "tic206b_reactor_temp_c",
    "TIC-607B\n反應釜油溫 (℃)": "tic607b_reactor_oil_temp_c",
    "TIC-607B\n冷卻水閥開度 (%)": "tic607b_cooling_valve_opening_pct",
    "TIC-206C\n反應釜溫度 (℃)": "tic206c_reactor_temp_c",
    "TIC-607C\n反應釜油溫 (℃)": "tic607c_reactor_oil_temp_c",
    "TIC-607C\n冷卻水閥開度 (%)": "tic607c_cooling_valve_opening_pct",
    "TIC-206D\n反應釜溫度 (℃)": "tic206d_reactor_temp_c",
    "TIC-607D\n反應釜油溫 (℃)": "tic607d_reactor_oil_temp_c",
    "TIC-607D\n冷卻水閥開度 (%)": "tic607d_cooling_valve_opening_pct",
    "A線冷凝回收溫度 (℃)": "line_a_condensate_recovery_temp_c",
    "B線冷凝回收溫度 (℃)": "line_b_condensate_recovery_temp_c",
    "C線冷凝回收溫度 (℃)": "line_c_condensate_recovery_temp_c",
    "D線冷凝回收溫度 (℃)": "line_d_condensate_recovery_temp_c",
    "A線回收液冷凝溫度 (℃)": "line_a_recovery_condensate_temp_c",
    "B線回收液冷凝溫度 (℃)": "line_b_recovery_condensate_temp_c",
    "C線回收液冷凝溫度 (℃)": "line_c_recovery_condensate_temp_c",
    "D線回收液冷凝溫度 (℃)": "line_d_recovery_condensate_temp_c",
    "RF-501A電流值 (A)": "rf501a_current_a",
    "RF-501B電流值 (A)": "rf501b_current_a",
    "RF-501C電流值 (A)": "rf501c_current_a",
    "RF-501A冷凍水流量指示 (M3/HR)": "rf501a_chilled_flow_m3_hr",
    "RF-501B冷凍水流量指示 (M3/HR)": "rf501b_chilled_flow_m3_hr",
    "RF-501C冷凍水流量指示 (M3/HR)": "rf501c_chilled_flow_m3_hr",
    "E-401A冷卻水流量計 (M3/HR)": "e401a_cooling_flow_m3_hr",
    "E-401B冷卻水流量計 (M3/HR)": "e401b_cooling_flow_m3_hr",
    "E-401C冷卻水流量計 (M3/HR)": "e401c_cooling_flow_m3_hr",
    "E-401D冷卻水流量計 (M3/HR)": "e401d_cooling_flow_m3_hr",
    "E-401A冷卻水出口溫度 (℃)": "e401a_cooling_outlet_temp_c",
    "E-401B冷卻水出口溫度 (℃)": "e401b_cooling_outlet_temp_c",
    "E-401C冷卻水出口溫度 (℃)": "e401c_cooling_outlet_temp_c",
    "E-401D冷卻水出口溫度 (℃)": "e401d_cooling_outlet_temp_c",
    "冷卻水塔開度值 (%)": "cooling_tower_opening_pct",
    "冷凍水壓差 (KG/CM2)": "chilled_water_pressure_diff_kg_cm2",
    "RF-501A COP": "rf501a_cop",
    "RF-501B COP": "rf501b_cop",
    "RF-501C COP": "rf501c_cop",
}


__all__ = [
    # 專案路徑配置
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "SRC_DIR",
    "MODELS_DIR",
    # 模型訓練參數
    "TEST_SIZE",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "DROPOUT",
    "RANDOM_STATE",
    "N_ESTIMATORS",
    "POWER_MODEL_PATH",
    "TEMPERATURE_MODEL_PATH",
    "COP_MODEL_PATH",
    "UNIFIED_FEATURES",
    "CHILLER_BOUNDS",
    # 控制變數分類
    "VALVE_CONTROL_VARIABLES",
    "COOLING_TOWER_CONTROL_VARIABLES",
    "COOLER_CONTROL_VARIABLES",
    "PUMP_CONTROL_VARIABLES",
    "FAN_CONTROL_VARIABLES",
    "ALL_CONTROL_VARIABLES",
    # 外部環境變數
    "WEATHER_VARIABLES",
    "PROCESS_DEMAND_VARIABLES",
    "ALL_EXTERNAL_VARIABLES",
    # 目標變數分類
    "PRIMARY_ENERGY_TARGETS",
    # 已移除 SECONDARY_ENERGY_TARGETS (舊版本殘留) => 不再匯出
    "EFFICIENCY_TARGETS",
    "BALANCE_TARGETS",
    "ALL_TARGET_VARIABLES",
    # 監控變數分類
    "TEMPERATURE_MONITORING_VARIABLES",
    "FLOW_MONITORING_VARIABLES",
    "PRESSURE_MONITORING_VARIABLES",
    "CURRENT_MONITORING_VARIABLES",
    "ALL_MONITORING_VARIABLES",
    # 衍生特徵變數
    "TEMPERATURE_DIFF_FEATURES",
    # 移除不存在的 COOLING_CAPACITY_FEATURES / RATIO_FEATURES / SYSTEM_STATISTICS_FEATURES
    "LOAD_DEMAND_FEATURES",
    # (上述特徵集合若未來需要可再定義)
    "TEMPERATURE_MODEL_BASE_FEATURES",
    "ALL_DERIVED_FEATURES",
    # 最佳化參數
    "LOAD_RATE_PERCENTILE",
    "OPERATION_THRESHOLD",
    "MAX_LOAD_RATE",
    "MIN_LOAD_RATE",
    "OPTIMIZATION_TARGETS",
    "DECISION_VARIABLES",
    "OPERATIONAL_CONSTRAINTS",
    "TIME_COLUMN",
    # 欄位對應表
    "COLUMN_MAPPING",
]
