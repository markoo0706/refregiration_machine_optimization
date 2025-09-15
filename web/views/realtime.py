"""
即時監控頁面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

from utils.autorefresh import st_autorefresh, show_refresh_status
from utils.cache import get_cached_data, clear_cache
from utils.data_generator import generate_mock_realtime_data


def show_realtime_dashboard():
    """即時監控儀表板頁面"""
    st.header("冰水主機即時監控儀表板")

    # 自動刷新控制
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader("關鍵系統指標")
    with col2:
        auto_refresh = st.checkbox("自動刷新", value=False, help="建議關閉以免影響操作")
    with col3:
        refresh_interval = st.selectbox("刷新間隔", [5, 10, 30, 60], index=1)
    with col4:
        if st.button("🔄 刷新數據", key="manual_refresh_data"):
            # 清除緩存強制重新生成數據
            clear_cache("realtime_data")
            st.rerun()

    # 自動刷新邏輯
    if auto_refresh:
        st_autorefresh(interval=refresh_interval, key="realtime_autorefresh")

    # 顯示刷新狀態
    show_refresh_status(refresh_interval, auto_refresh, "realtime_autorefresh")

    # 生成數據 - 使用緩存機制
    df_realtime = get_cached_data(
        "realtime_data",
        generate_mock_realtime_data,
        cache_duration=10  # 10秒緩存
    )
    current_data = df_realtime.iloc[-1].to_dict()

    # 顯示當前時間
    st.caption(f"最後更新時間: {current_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # 第一排：核心指標
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_power = float(current_data["cooling_system_power"])
        prev_power = float(df_realtime["cooling_system_power"].iloc[-2])
        power_delta = current_power - prev_power

        st.metric(
            label="冷卻系統功耗",
            value=f"{current_power:.1f} kW",
            delta=f"{power_delta:+.1f} kW"
        )

    with col2:
        current_cop = float(current_data["cooling_system_cop"])
        prev_cop = float(df_realtime["cooling_system_cop"].iloc[-2])
        cop_delta = current_cop - prev_cop

        st.metric(
            label="系統 COP",
            value=f"{current_cop:.2f}",
            delta=f"{cop_delta:+.2f}"
        )

    with col3:
        current_temp = float(current_data["cooling_tower_temp"])
        prev_temp = float(df_realtime["cooling_tower_temp"].iloc[-2])
        temp_delta = current_temp - prev_temp

        st.metric(
            label="冷卻水溫",
            value=f"{current_temp:.1f} °C",
            delta=f"{temp_delta:+.1f} °C"
        )

    with col4:
        current_load = float(current_data["fan_load_rate"])
        prev_load = float(df_realtime["fan_load_rate"].iloc[-2])
        load_delta = current_load - prev_load

        st.metric(
            label="風扇負載率",
            value=f"{current_load:.1f} %",
            delta=f"{load_delta:+.1f} %"
        )

    # 第二排：趨勢圖表
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("功耗與效率趨勢")

        # 創建雙軸圖表
        fig_power = make_subplots(specs=[[{"secondary_y": True}]])

        # 功耗趨勢
        fig_power.add_trace(
            go.Scatter(
                x=df_realtime["timestamp"],
                y=df_realtime["cooling_system_power"],
                mode="lines+markers",
                name="冷卻功耗 (kW)",
                line=dict(color="red", width=2)
            ),
            secondary_y=False,
        )

        # COP 趨勢
        fig_power.add_trace(
            go.Scatter(
                x=df_realtime["timestamp"],
                y=df_realtime["cooling_system_cop"],
                mode="lines+markers",
                name="COP",
                line=dict(color="blue", width=2)
            ),
            secondary_y=True,
        )

        fig_power.update_xaxes(title_text="時間")
        fig_power.update_yaxes(title_text="功耗 (kW)", secondary_y=False)
        fig_power.update_yaxes(title_text="COP", secondary_y=True)
        fig_power.update_layout(height=400, showlegend=True)

        st.plotly_chart(fig_power, use_container_width=True)

    with col2:
        st.subheader("溫度與負載趨勢")

        fig_temp = make_subplots(specs=[[{"secondary_y": True}]])

        # 溫度趨勢
        fig_temp.add_trace(
            go.Scatter(
                x=df_realtime["timestamp"],
                y=df_realtime["cooling_tower_temp"],
                mode="lines+markers",
                name="冷卻水溫 (°C)",
                line=dict(color="orange", width=2)
            ),
            secondary_y=False,
        )

        fig_temp.add_trace(
            go.Scatter(
                x=df_realtime["timestamp"],
                y=df_realtime["ambient_temp"],
                mode="lines",
                name="環境溫度 (°C)",
                line=dict(color="lightblue", width=1, dash="dash")
            ),
            secondary_y=False,
        )

        # 負載率趨勢
        fig_temp.add_trace(
            go.Scatter(
                x=df_realtime["timestamp"],
                y=df_realtime["fan_load_rate"],
                mode="lines+markers",
                name="風扇負載 (%)",
                line=dict(color="green", width=2)
            ),
            secondary_y=True,
        )

        fig_temp.update_xaxes(title_text="時間")
        fig_temp.update_yaxes(title_text="溫度 (°C)", secondary_y=False)
        fig_temp.update_yaxes(title_text="負載率 (%)", secondary_y=True)
        fig_temp.update_layout(height=400, showlegend=True)

        st.plotly_chart(fig_temp, use_container_width=True)

    # 第三排：系統狀態摘要
    st.markdown("---")
    st.subheader("系統狀態摘要")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 效率狀態
        efficiency = float(current_data["system_efficiency"])
        if efficiency > 90:
            status_color = "🟢"
            status_text = "優秀"
        elif efficiency > 80:
            status_color = "🟡"
            status_text = "良好"
        else:
            status_color = "🔴"
            status_text = "需改善"

        st.metric("系統效率", f"{efficiency:.1f}%", help=f"狀態: {status_color} {status_text}")

    with col2:
        # 散熱能力
        heat_rejection = float(current_data["heat_rejection"])
        st.metric("散熱能力", f"{heat_rejection:.0f} kW")

    with col3:
        # 運行時間（模擬）
        uptime_hours = 24 * 7 + random.randint(1, 23)  # 模擬運行一週多
        st.metric("連續運行時間", f"{uptime_hours} 小時")

    # 警示與建議
    st.markdown("---")
    st.subheader("系統建議")

    warnings = []
    recommendations = []

    current_power_val = current_power
    current_cop_val = current_cop
    current_temp_val = current_temp
    current_load_val = current_load

    if current_power_val > 200:
        warnings.append("功耗較高，建議檢查系統負載")
    if current_cop_val < 2.5:
        warnings.append("COP偏低，建議檢查系統效率")
    if current_temp_val > 35:
        warnings.append("冷卻水溫偏高，建議檢查散熱系統")
    if current_load_val > 90:
        warnings.append("風扇負載過高，建議分散負載")

    if current_cop_val > 3.5:
        recommendations.append("系統效率良好，可考慮適度增加負載")
    if current_temp_val < 30:
        recommendations.append("冷卻效果良好，系統運行穩定")

    if warnings:
        st.warning("注意事項:")
        for warning in warnings:
            st.write(f"⚠️ {warning}")

    if recommendations:
        st.success("系統建議:")
        for rec in recommendations:
            st.write(f"💡 {rec}")

    if not warnings and not recommendations:
        st.info("📊 系統運行正常，所有指標都在正常範圍內")