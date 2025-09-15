"""
模擬數據生成工具
"""
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta


def generate_mock_realtime_data() -> pd.DataFrame:
    """生成模擬的即時監控數據"""
    # 使用當前時間作為隨機種子，確保每次都有不同的數據
    random.seed(int(time.time()))

    # 生成過去24小時的數據
    now = datetime.now()
    time_points = [now - timedelta(hours=i) for i in range(24, 0, -1)]

    # 基準值和變化範圍
    base_values = {
        "cooling_system_power": 180,  # kW
        "cooling_system_cop": 3.2,
        "cooling_tower_temp": 28,     # °C
        "fan_load_rate": 65,         # %
        "system_efficiency": 85,     # %
        "ambient_temp": 25,          # °C
        "heat_rejection": 580,       # kW
    }

    data = []
    for timestamp in time_points:
        # 添加時間相關的變化（白天高，晚上低）
        hour_factor = 0.8 + 0.4 * abs(np.sin((timestamp.hour - 6) * np.pi / 12))

        # 添加隨機波動
        noise = random.uniform(-0.1, 0.1)

        record = {
            "timestamp": timestamp,
            "cooling_system_power": base_values["cooling_system_power"] * hour_factor * (1 + noise),
            "cooling_system_cop": base_values["cooling_system_cop"] * (1.2 - 0.2 * hour_factor) * (1 + noise * 0.5),
            "cooling_tower_temp": base_values["cooling_tower_temp"] + 5 * hour_factor + random.uniform(-2, 2),
            "fan_load_rate": base_values["fan_load_rate"] * hour_factor + random.uniform(-10, 10),
            "system_efficiency": base_values["system_efficiency"] * (1.1 - 0.1 * hour_factor) * (1 + noise * 0.3),
            "ambient_temp": base_values["ambient_temp"] + 8 * hour_factor + random.uniform(-3, 3),
            "heat_rejection": base_values["heat_rejection"] * hour_factor * (1 + noise * 0.8),
        }
        data.append(record)

    return pd.DataFrame(data)


def create_inline_sample_data() -> pd.DataFrame:
    """創建內嵌示例數據"""
    n_samples = 200
    start_time = datetime.now() - timedelta(hours=50)

    # 創建時間序列
    time_index = [start_time + timedelta(minutes=15*i) for i in range(n_samples)]

    # 生成示例數據
    np.random.seed(42)
    data = {
        'time': time_index,
        'ambient_temperature_c': 25 + 10 * np.random.random(n_samples),
        'ambient_humidity_rh': 50 + 30 * np.random.random(n_samples),
        'cooling_tower_opening_pct': 30 + 50 * np.random.random(n_samples),
        'cooling_tower_outlet_temp_c': 25 + 15 * np.random.random(n_samples),
        'cooling_tower_return_temp_c': 30 + 15 * np.random.random(n_samples),
        'fan_510a_power_kw': 50 + 100 * np.random.random(n_samples),
        'fan_510b_power_kw': 45 + 95 * np.random.random(n_samples),
        'fan_510c_power_kw': 48 + 98 * np.random.random(n_samples),
        'cooling_pump_g511a_power_kw': 20 + 40 * np.random.random(n_samples),
        'cooling_pump_g511b_power_kw': 18 + 38 * np.random.random(n_samples),
        'fan_510a_current_a': 30 + 70 * np.random.random(n_samples),
        'fan_510b_current_a': 28 + 68 * np.random.random(n_samples),
        'fan_510c_current_a': 29 + 69 * np.random.random(n_samples),
    }

    return pd.DataFrame(data)


def generate_performance_data() -> pd.DataFrame:
    """生成效能監控數據"""
    dates = pd.date_range(start="2025-09-01", end="2025-09-14", freq="D")

    performance_data = {
        "date": dates,
        "power_consumption": np.random.uniform(180, 320, len(dates)),
        "efficiency": np.random.uniform(2.5, 4.5, len(dates)),
        "temperature": np.random.uniform(6.5, 8.0, len(dates)),
    }

    return pd.DataFrame(performance_data)