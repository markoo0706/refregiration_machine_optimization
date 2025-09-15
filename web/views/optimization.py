"""
多目標最佳化頁面
"""
import streamlit as st
import time
from utils.api import api_request


def check_task_status(task_id: str):
    """檢查任務狀態並顯示結果"""
    if not task_id:
        return

    progress_placeholder = st.empty()

    for _ in range(60):  # Increase timeout to 2 minutes
        response = api_request(f"/logs/{task_id}")

        if response:
            status = response.get('status', 'UNKNOWN')

            with progress_placeholder.container():
                if status == 'PENDING':
                    st.info("任務等待中...")
                elif status == 'PROGRESS' or status == 'STARTED':
                    st.info("任務執行中...")
                elif status == 'SUCCESS':
                    st.success("最佳化完成！")
                    result = response.get('result', {})
                    if result and 'results' in result and 'solutions' in result['results']:
                        import pandas as pd
                        import plotly.express as px

                        st.subheader("最佳化結果")
                        solutions = result['results']['solutions']
                        df = pd.DataFrame(solutions)
                        
                        st.dataframe(df)

                        st.subheader("Pareto Front")
                        fig = px.scatter(df, x="power_consumption", y="cop", title="Pareto Front: Power vs. COP")
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("任務成功，但沒有回傳有效的結果。")
                        st.json(result)
                    break
                elif status == 'FAILURE':
                    st.error("任務失敗")
                    st.text(str(response.get('info', '')))
                    break
                else:
                    st.warning(f"任務狀態: {status}")

        time.sleep(2)


def show_optimization():
    """多目標最佳化頁面"""
    st.header("多目標最佳化")

    with st.form("optimization_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            target_temp = st.number_input("目標溫度 (°C)", 5.0, 10.0, 7.0, 0.1)
            algorithm = st.selectbox("最佳化演算法",
                                   ["nsga2", "moea_d", "spea2", "particle_swarm"],
                                   index=0)
            objective = st.selectbox("最佳化目標",
                                     ["total_power", "weighted_cop", "operational_cost", "load_balance_score"],
                                     index=0)

        with col2:
            max_iterations = st.slider("最大迭代次數", 50, 500, 100)
            population_size = st.slider("族群大小", 20, 200, 100)

        with col3:
            weight_power = st.slider("功耗權重", 0.1, 2.0, 1.0, 0.1)
            weight_efficiency = st.slider("效率權重", 0.1, 2.0, 0.8, 0.1)

        submit_button = st.form_submit_button("🚀 開始最佳化", type="primary")

    if submit_button:
        # Consolidate all parameters into a single dictionary
        optimization_params = {
            "objective": objective,
            "target_temp": target_temp,
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "population_size": population_size,
            "weight_power": weight_power,
            "weight_efficiency": weight_efficiency,
            # You can add other fixed parameters here, e.g., from other UI inputs
            # "ambient_temp": 30.0, 
            # "ambient_humidity": 75.0
        }

        with st.spinner("正在提交最佳化任務..."):
            # Call the Celery task via the API endpoint
            response = api_request("/optimize", method="POST", data=optimization_params)

        if response and response.get('task_id'):
            task_id = response.get('task_id')
            st.success(f"✅ 最佳化任務已成功提交！任務 ID: {task_id}")
            
            with st.expander("查看即時進度與結果", expanded=True):
                check_task_status(task_id)
        else:
            st.error(f"任務提交失敗: {response}")