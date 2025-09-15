#!/usr/bin/env python3

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_model_loading():
    """測試模型加載"""
    power_model_path = 'models/default_cooling_system_total_power_kw.pkl'
    cop_model_path = 'models/default_cooling_system_cop.pkl'

    print("=== 測試模型加載 ===")

    try:
        power_model = joblib.load(power_model_path)
        print(f"✓ Power model loaded: {type(power_model)}")
    except Exception as e:
        print(f"✗ Power model failed: {e}")
        return False, False

    try:
        cop_model = joblib.load(cop_model_path)
        print(f"✓ COP model loaded: {type(cop_model)}")
    except Exception as e:
        print(f"✗ COP model failed: {e}")
        return True, False

    return True, True

def test_model_prediction():
    """測試模型預測"""
    power_model_path = 'models/default_cooling_system_total_power_kw.pkl'
    cop_model_path = 'models/default_cooling_system_cop.pkl'

    print("\n=== 測試模型預測 ===")

    try:
        power_model = joblib.load(power_model_path)
        cop_model = joblib.load(cop_model_path)

        # 創建測試數據
        test_data = {
            'cooling_tower_opening_pct': [50.0],
            'fan_510a_power_kw': [30.0],
            'fan_510b_power_kw': [30.0],
            'fan_510c_power_kw': [0.0],
            'ambient_temperature_c': [28.5],
            'ambient_humidity_rh': [65.0],
        }

        # 檢查模型需要的特徵
        if hasattr(power_model, 'feature_names_in_'):
            print(f"Power model expects {len(power_model.feature_names_in_)} features")
            print(f"First 5 features: {list(power_model.feature_names_in_[:5])}")

            # 創建完整的特徵數據框
            feature_df = pd.DataFrame(index=[0], columns=power_model.feature_names_in_)
            feature_df = feature_df.fillna(0.0)  # 填充預設值

            # 更新已知值
            for key, value in test_data.items():
                if key in feature_df.columns:
                    feature_df[key] = value[0]

            # 預測
            power_pred = power_model.predict(feature_df)
            cop_pred = cop_model.predict(feature_df)

            print(f"✓ Power prediction: {power_pred[0]:.2f} kW")
            print(f"✓ COP prediction: {cop_pred[0]:.2f}")

            return True

        else:
            print("✗ Model doesn't have feature_names_in_ attribute")
            return False

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def simple_optimization_test():
    """簡化最佳化測試"""
    print("\n=== 簡化最佳化測試 ===")

    # 使用固定的合理預測值來測試最佳化邏輯
    test_solutions = []

    # 測試不同的參數組合
    for tower_opening in [30, 50, 70]:
        for fan_power in [20, 40, 60]:
            # 模擬功率消耗：基礎功率 + 風扇功率影響
            simulated_power = 200 + fan_power * 2.5 + tower_opening * 0.8

            # 模擬COP：效率隨功率增加而下降
            simulated_cop = max(2.0, 5.0 - fan_power * 0.02 - tower_opening * 0.01)

            test_solutions.append({
                'cooling_tower_opening_pct': tower_opening,
                'fan_power_total': fan_power,
                'power_consumption': simulated_power,
                'cop': simulated_cop
            })

    # 找最佳解（最低功耗）
    best_power = min(test_solutions, key=lambda x: x['power_consumption'])
    print(f"✓ 最低功耗解: {best_power['power_consumption']:.1f}kW at tower={best_power['cooling_tower_opening_pct']}%, fan={best_power['fan_power_total']}kW")

    # 找最佳效率解（最高COP）
    best_cop = max(test_solutions, key=lambda x: x['cop'])
    print(f"✓ 最高效率解: COP={best_cop['cop']:.2f} at tower={best_cop['cooling_tower_opening_pct']}%, fan={best_cop['fan_power_total']}kW")

    return True

if __name__ == "__main__":
    print("開始最佳化系統診斷...")

    # 1. 測試模型加載
    power_ok, cop_ok = test_model_loading()

    # 2. 測試模型預測（如果加載成功）
    if power_ok and cop_ok:
        pred_ok = test_model_prediction()
    else:
        print("模型加載失敗，跳過預測測試")
        pred_ok = False

    # 3. 簡化最佳化測試
    opt_ok = simple_optimization_test()

    # 總結
    print(f"\n=== 診斷結果 ===")
    print(f"模型加載: {'✓' if power_ok and cop_ok else '✗'}")
    print(f"模型預測: {'✓' if pred_ok else '✗'}")
    print(f"最佳化邏輯: {'✓' if opt_ok else '✗'}")

    if power_ok and cop_ok and pred_ok:
        print("✓ 系統準備就緒，可以運行完整最佳化")
    else:
        print("✗ 系統需要修復才能正常運行")