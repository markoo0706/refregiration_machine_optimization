"""
系統監控頁面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

from utils.api import api_request, check_api_health
from utils.data_generator import generate_performance_data


def check_task_status(task_id: str):
    """檢查任務狀態"""
    if not task_id:
        return

    progress_placeholder = st.empty()

    for _ in range(30):  # 最多檢查30次
        response = api_request(f"/logs/{task_id}")

        if response:
            status = response.get('status', 'UNKNOWN')

            with progress_placeholder.container():
                if status == 'PENDING':
                    st.info("任務等待中...")
                elif status == 'PROGRESS':
                    st.info("任務執行中...")
                elif status == 'SUCCESS':
                    st.success("任務完成！")
                    result = response.get('result', {})
                    if result:
                        st.json(result)
                    break
                elif status == 'FAILURE':
                    st.error("任務失敗")
                    st.text(str(response.get('info', '')))
                    break
                else:
                    st.warning(f"任務狀態: {status}")

        time.sleep(2)

    progress_placeholder.empty()


def show_performance_charts():
    """顯示效能圖表"""
    try:
        # 使用模組化的數據生成器
        df = generate_performance_data()
    except ImportError:
        # 如果無法導入，使用原始方法
        dates = pd.date_range(start="2025-09-01", end="2025-09-14", freq="D")
        performance_data = {
            "date": dates,
            "power_consumption": np.random.uniform(180, 320, len(dates)),
            "efficiency": np.random.uniform(2.5, 4.5, len(dates)),
            "temperature": np.random.uniform(6.5, 8.0, len(dates)),
        }
        df = pd.DataFrame(performance_data)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(df, x="date", y="power_consumption",
                      title="功耗趨勢", labels={"power_consumption": "功耗 (kW)"})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(df, x="date", y="efficiency",
                      title="效率趨勢", labels={"efficiency": "COP"})
        st.plotly_chart(fig2, use_container_width=True)

    # 綜合儀表板
    fig3 = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )

    current_power = df["power_consumption"].iloc[-1]
    current_efficiency = df["efficiency"].iloc[-1]
    current_temp = df["temperature"].iloc[-1]

    fig3.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_power,
        gauge={"axis": {"range": [0, 400]}, "bar": {"color": "darkblue"}},
    ), row=1, col=1)

    fig3.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_efficiency,
        gauge={"axis": {"range": [0, 6]}, "bar": {"color": "green"}},
    ), row=1, col=2)

    fig3.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_temp,
        gauge={"axis": {"range": [0, 15]}, "bar": {"color": "red"}},
    ), row=1, col=3)

    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)


def show_api_docs():
    """API文件"""
    st.header("API文件")

    st.subheader("可用端點")

    endpoints = [
        {"端點": "/health", "方法": "GET", "說明": "健康檢查"},
        {"端點": "/upload-data", "方法": "POST", "說明": "上傳數據"},
        {"端點": "/train/temperature", "方法": "POST", "說明": "訓練溫度模型"},
        {"端點": "/train/power", "方法": "POST", "說明": "訓練功耗模型"},
        {"端點": "/optimize", "方法": "POST", "說明": "執行最佳化"},
        {"端點": "/logs/{task_id}", "方法": "GET", "說明": "查看任務日誌"},
    ]

    df = pd.DataFrame(endpoints)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("使用範例")

    API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

    st.code(f"""
# 健康檢查
curl {API_BASE}/health

# 訓練模型
curl -X POST {API_BASE}/train/temperature

# 執行最佳化
curl -X POST {API_BASE}/optimize \
  -H "Content-Type: application/json" \
  -d '{{"target_temp": 7.0, "algorithm": "nsga2"}}'
""", language="bash")


def show_monitoring():
    """系統監控頁面"""
    st.header("系統監控")

    # 系統狀態
    col1, col2, col3 = st.columns(3)

    health_check = api_request("/health")
    with col1:
        if health_check:
            status = "正常"
            st.metric("API狀態", status)
        else:
            status = "離線"
            st.metric("API狀態", status, help="API 服務器未啟動，顯示模擬數據")
            st.warning("API 服務器未啟動，請執行 `python api.py` 啟動後端服務")

    with col2:
        st.metric("系統版本", "v1.0.0")

    with col3:
        st.metric("運行時間", "計算中...")

    # API 連線測試
    if st.button("測試 API 連線"):
        if check_api_health():
            st.success("✅ API 連線成功！")
        else:
            st.error("❌ API 連線失敗！")

    st.markdown("---")

    # 任務監控
    st.subheader("任務監控")

    task_id = st.text_input("輸入任務ID查看狀態")
    if task_id and st.button("查看任務狀態"):
        check_task_status(task_id)

    # 效能監控圖表
    if st.button("生成效能報告"):
        show_performance_charts()

    # API 文件區塊
    st.markdown("---")
    show_api_docs()