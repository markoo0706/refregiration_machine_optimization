#!/usr/bin/env python3
import numpy as np

def test_eval_function():
    """測試_eval函數的數學邏輯"""
    # 模擬幾個測試參數組合
    test_cases = [
        [50.0, 10.0, 10.0, 0.0],  # 正常情況
        [0.0, 0.0, 0.0, 0.0],     # 最小值
        [100.0, 160.0, 160.0, 160.0],  # 最大值
        [25.0, 5.0, 15.0, 8.0],   # 隨機情況
    ]

    for i, x in enumerate(test_cases):
        print(f"\n=== 測試案例 {i+1}: {x} ===")

        # 複製_eval函數的邏輯
        tower_opening = max(0.0, min(100.0, float(x[0])))
        fan_a = max(0.0, min(160.0, float(x[1])))
        fan_b = max(0.0, min(160.0, float(x[2])))
        fan_c = max(0.0, min(160.0, float(x[3])))

        print(f"輸入: tower={tower_opening}, fans=[{fan_a}, {fan_b}, {fan_c}]")

        # 功率計算
        base_power = 530.0
        fan_total = fan_a + fan_b + fan_c
        tower_factor = 1.0 + (50.0 - tower_opening) * 0.002
        total_power = (base_power + fan_total) * tower_factor
        power = np.clip(total_power, 300.0, 800.0)

        print(f"功率計算: base={base_power}, fan_total={fan_total}")
        print(f"         tower_factor={tower_factor}, total={total_power}")
        print(f"         final_power={power}")

        # COP計算
        base_cop = 4.2
        optimal_fan = 15.0
        fan_efficiency = 1.0 - abs(fan_total - optimal_fan) * 0.01
        fan_efficiency = np.clip(fan_efficiency, 0.8, 1.2)

        tower_efficiency = 0.7 + (tower_opening / 100.0) * 0.4
        tower_efficiency = np.clip(tower_efficiency, 0.7, 1.1)

        cop = base_cop * fan_efficiency * tower_efficiency
        cop = np.clip(cop, 2.5, 6.0)

        print(f"COP計算: base={base_cop}, fan_eff={fan_efficiency}")
        print(f"        tower_eff={tower_efficiency}, final_cop={cop}")

        # 檢查有效性
        if not np.isfinite(power):
            print(f"❌ 功率無效: {power}")
        if not np.isfinite(cop):
            print(f"❌ COP無效: {cop}")

        power_output = max(250.0, min(700.0, float(power)))
        cop_output = max(-8.0, min(-1.5, float(-cop)))

        print(f"最終輸出: F=[{power_output}, {cop_output}]")

        if np.isfinite(power_output) and np.isfinite(cop_output):
            print("✅ 此解有效")
        else:
            print("❌ 此解無效")

if __name__ == "__main__":
    test_eval_function()