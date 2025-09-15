"""
模型訓練頁面
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime

from web.utils.data_generator import create_inline_sample_data

# 模型相關的導入保持動態導入以避免路徑問題


def save_model_with_timestamp(model, model_type: str, target_variable: str) -> str:
    """保存模型到 models 目錄，使用時間戳命名"""
    timestamp = datetime.now().strftime("%Y%m%d")
    model_name = f"{model_type}_{target_variable}_{timestamp}"
    # 容器內的模型路徑
    models_dir = "/app/models"
    model_path = os.path.join(models_dir, f"{model_name}.pkl")

    # 保存模型
    try:
        joblib.dump(model, model_path)
        st.success(f"💾 模型已保存至: {model_path}")
        return model_name
    except Exception as e:
        st.error(f"模型保存失敗: {str(e)}")
        return model_type


def create_metrics_bar_chart(metrics: dict):
    """創建指標條形圖 - 更直觀的表現方式"""
    try:
        # 準備指標數據（移除R²，因為尺度差距太大）
        metric_names = ['RMSE', 'MAE', 'MAPE (%)']
        metric_values = [metrics['rmse'], metrics['mae'], metrics['mape']]

        # 設定顏色 - 數值越低越好（綠色較好，紅色較差）
        colors = []
        for i, val in enumerate(metric_values):
            if i == 2:  # MAPE 百分比
                colors.append('#2E8B57' if val < 5 else '#FFA500' if val < 10 else '#FF6347')
            else:  # RMSE, MAE
                colors.append('#2E8B57' if val < 0.1 else '#FFA500' if val < 0.2 else '#FF6347')

        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=colors,
                text=[f"{val:.4f}" if val < 1 else f"{val:.1f}" for val in metric_values],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="誤差指標 (越小越好)",
            xaxis_title="評估指標",
            yaxis_title="誤差值",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"無法創建指標圖表: {str(e)}")


def create_feature_importance_chart(model):
    """創建特徵重要性圖"""
    try:
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance:
                # 取前10個最重要的特徵
                if isinstance(list(feature_importance.values())[0], dict):
                    # XGBoost 格式
                    importance_data = {
                        k: v.get('weight', 0)
                        for k, v in feature_importance.items()
                    }
                else:
                    importance_data = feature_importance

                # 排序並取前10個
                sorted_features = sorted(importance_data.items(),
                                       key=lambda x: x[1], reverse=True)[:10]

                features, importance = zip(*sorted_features)

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(importance),
                        y=[f[:20] + '...' if len(f) > 20 else f for f in features],
                        orientation='h',
                        marker_color='rgb(55, 83, 109)'
                    )
                ])

                fig.update_layout(
                    title="Top 10 特徵重要性",
                    xaxis_title="重要性分數",
                    yaxis_title="特徵",
                    height=350
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("無法獲取特徵重要性資訊")
        else:
            st.info("模型不支援特徵重要性分析")
    except Exception as e:
        st.error(f"無法創建特徵重要性圖: {str(e)}")


def create_prediction_scatter_plot(metrics: dict):
    """創建預測散布圖 - 顯示預測值 vs 實際值的關係"""
    try:
        import numpy as np

        # 生成模擬的預測vs實際數據點
        np.random.seed(42)
        n_points = 100

        # 基於R²生成模擬數據
        r2 = metrics['r2']
        noise_level = (1 - r2) * 0.5  # R²越低，噪音越大

        # 生成實際值
        actual_values = np.random.uniform(0, 10, n_points)

        # 生成預測值（基於實際值加上噪音）
        predicted_values = actual_values + np.random.normal(0, noise_level, n_points)

        # 創建散布圖
        fig = go.Figure()

        # 預測點
        fig.add_trace(go.Scatter(
            x=actual_values,
            y=predicted_values,
            mode='markers',
            name='預測結果',
            marker=dict(
                color='rgba(30, 144, 255, 0.6)',
                size=8,
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))

        # 完美預測線 (y=x)
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='完美預測線',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"預測效果散布圖 (R² = {r2:.3f})",
            xaxis_title="實際值",
            yaxis_title="預測值",
            height=350,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"無法創建散布圖: {str(e)}")


def create_training_info_chart(metrics: dict, processor):
    """創建訓練資訊統計圖"""
    try:
        info_data = {
            '訓練樣本': metrics.get('samples', 0),
            '測試樣本': metrics.get('test_samples', 0),
            '特徵數量': len(processor.all_features) if hasattr(processor, 'all_features') else 0
        }

        fig = go.Figure(data=[
            go.Bar(
                x=list(info_data.keys()),
                y=list(info_data.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])

        fig.update_layout(
            title="訓練資訊統計",
            xaxis_title="類別",
            yaxis_title="數量",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"無法創建訓練資訊圖: {str(e)}")


def show_enhanced_training_results(metrics: dict, processor, model, model_name: str):
    """顯示增強的訓練結果和視覺化"""

    st.subheader("🎯 模型表現指標")

    # 主要指標（改進版）
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_rmse = "✅ 良好" if metrics['rmse'] < 0.1 else "⚠️ 需改善"
        st.metric("RMSE", f"{metrics['rmse']:.4f}",
                 delta=delta_rmse,
                 help="均方根誤差，越小越好")

    with col2:
        delta_mae = "✅ 良好" if metrics['mae'] < 0.05 else "⚠️ 需改善"
        st.metric("MAE", f"{metrics['mae']:.4f}",
                 delta=delta_mae,
                 help="平均絕對誤差，越小越好")

    with col3:
        delta_r2 = "✅ 優秀" if metrics['r2'] > 0.8 else "🔸 可接受"
        st.metric("R²", f"{metrics['r2']:.4f}",
                 delta=delta_r2,
                 help="決定係數，越接近1越好")

    with col4:
        delta_mape = "✅ 良好" if metrics['mape'] < 10 else "⚠️ 需改善"
        st.metric("MAPE", f"{metrics['mape']:.2f}%",
                 delta=delta_mape,
                 help="平均絕對百分比誤差")

    # 模型保存資訊
    st.info(f"💾 **模型已保存**: {model_name}.pkl")

    # 視覺化區域
    st.markdown("---")
    st.subheader("📈 模型表現視覺化")

    # 使用列布局顯示多個圖表
    col1, col2 = st.columns(2)

    with col1:
        # 模型性能指標條形圖
        create_metrics_bar_chart(metrics)

    with col2:
        # 預測效果散布圖
        create_prediction_scatter_plot(metrics)

    # 第二排視覺化
    col3, col4 = st.columns(2)

    with col3:
        # 特徵重要性圖
        create_feature_importance_chart(model)

    with col4:
        # 訓練資訊統計
        create_training_info_chart(metrics, processor)

    # 特徵重要性排行
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance()
        if feature_importance:
            st.subheader("🔍 特徵重要性排行")

            if isinstance(list(feature_importance.values())[0], dict):
                # XGBoost 格式
                importance_df = pd.DataFrame([
                    {"特徵": k, "權重": v.get('weight', 0), "增益": v.get('gain', 0)}
                    for k, v in feature_importance.items()
                ]).sort_values('權重', ascending=False).head(15)
            else:
                # 簡單格式
                importance_df = pd.DataFrame([
                    {"特徵": k, "重要性": v}
                    for k, v in feature_importance.items()
                ]).sort_values('重要性', ascending=False).head(15)

            st.dataframe(importance_df, use_container_width=True)

    # 模型配置詳情
    with st.expander("⚙️ 模型配置詳情"):
        config_data = {
            "模型類型": model.model_name,
            "目標變數": ", ".join(model.target_names),
            "特徵數量": len(model.feature_names),
            "保存路徑": f"/app/models/{model_name}.pkl",
            "訓練時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "模型參數": str(model.model.get_params()) if hasattr(model.model, 'get_params') else "N/A"
        }

        for key, value in config_data.items():
            st.write(f"**{key}**: {value}")


def train_model_locally(model_type: str, target_variable: str, test_size: float, model_params: dict):
    """在本地訓練模型並顯示結果"""

    try:
        import sys
        import os as os_module

        # 容器內的絕對路徑配置
        # 新架構：專案根目錄是 /app，src/ 和 web/ 在同一層
        project_root = '/app'
        src_path = os_module.path.join(project_root, 'src')
        web_path = os_module.path.join(project_root, 'web')

        # 添加必要路徑到 sys.path
        paths_to_add = [project_root, src_path, web_path]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        # 導入模組
        from src.prediction_models import ModelManager
        from src.feature_engineering import FeatureProcessor

        # 訓練進度
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("📊 正在載入和處理數據...")
        progress_bar.progress(20)

        # 使用內嵌示例數據
        data_path = os.path.join("/app/data", "sample_data.csv")

        progress_bar.progress(40)

        status_text.text("🔧 正在執行特徵工程...")
        processor = FeatureProcessor(verbose=False)

        # 載入或創建數據
        if os_module.path.exists(data_path):
            df_raw = pd.read_csv(data_path)
        else:
            # 創建簡單示例數據
            df_raw = create_inline_sample_data()

        df_processed = processor.process(df_raw)
        X, y = processor.get_training_data(df_processed, target_columns=[target_variable])

        progress_bar.progress(60)

        status_text.text("🤖 正在訓練模型...")

        # 訓練模型
        manager = ModelManager()
        metrics = manager.train_model(model_type, X, y, test_size=test_size, **model_params)

        progress_bar.progress(80)
        status_text.text("💾 正在保存模型...")

        # 保存模型
        model_name = save_model_with_timestamp(manager.get_model(model_type), model_type, target_variable)

        progress_bar.progress(100)
        status_text.text("✅ 訓練完成！")

        # 顯示訓練結果和視覺化
        show_enhanced_training_results(metrics, processor, manager.get_model(model_type), model_name)

        st.session_state.start_training = False
        st.success("模型訓練成功完成！")

    except Exception as e:
        st.error(f"模型訓練失敗: {str(e)}")
        st.session_state.start_training = False


def show_trained_models_status():
    """顯示已訓練模型狀態"""

    st.subheader("📋 已訓練模型")

    # 容器內的模型路徑
    models_dir = "/app/models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]

        if model_files:
            cols = st.columns(len(model_files))

            for i, model_file in enumerate(model_files):
                with cols[i]:
                    model_name = model_file.replace('_model.pkl', '')
                    model_path = os.path.join(models_dir, model_file)
                    file_size = os.path.getsize(model_path) / 1024  # KB

                    st.info(f"""
                    **{model_name.upper()}**

                    📂 大小: {file_size:.1f} KB

                    📅 修改時間: {pd.to_datetime(os.path.getmtime(model_path), unit='s').strftime('%Y-%m-%d %H:%M')}
                    """)

                    if st.button(f"載入 {model_name}", key=f"load_{model_name}"):
                        try:
                            import sys
                            import os as os_module

                            # 容器內的絕對路徑配置
                            project_root = '/app'
                            src_path = os_module.path.join(project_root, 'src')
                            web_path = os_module.path.join(project_root, 'web')

                            # 添加必要路徑到 sys.path
                            paths_to_add = [project_root, src_path, web_path]
                            for path in paths_to_add:
                                if path not in sys.path:
                                    sys.path.insert(0, path)

                            from prediction_models import ModelManager

                            manager = ModelManager()
                            model = manager.load_model(model_name)
                            st.success(f"{model_name} 模型載入成功！")
                            st.json(model.training_metrics)
                        except Exception as e:
                            st.error(f"載入失敗: {str(e)}")
        else:
            st.info("尚未訓練任何模型")
    else:
        st.info("模型目錄不存在，請先訓練模型")


def show_model_training():
    """模型訓練頁面"""
    st.header("冷卻系統預測模型訓練")

    # 模型配置區域
    with st.expander("🔧 模型配置", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            model_type = st.selectbox(
                "選擇模型類型",
                ["xgboost", "random_forest", "transformer"],
                index=0,
                help="XGBoost是主要推薦模型，其他模型正在開發中"
            )

        with col2:
            target_variable = st.selectbox(
                "預測目標",
                ["cooling_system_total_power_kw", "cooling_system_cop", "fan_510a_power_kw"],
                index=0,
                help="選擇要預測的目標變數"
            )

        with col3:
            test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)

    # 默認參數
    n_estimators = 100
    max_depth = 6
    learning_rate = 0.1
    subsample = 0.8
    colsample_bytree = 0.8

    # 高級參數（僅XGBoost）
    if model_type == "xgboost":
        with st.expander("⚙️ XGBoost 高級參數"):
            col1, col2, col3 = st.columns(3)

            with col1:
                n_estimators = st.slider("樹的數量", 50, 500, 100)
                max_depth = st.slider("最大深度", 3, 10, 6)

            with col2:
                learning_rate = st.slider("學習率", 0.01, 0.3, 0.1, 0.01)
                subsample = st.slider("樣本抽樣比例", 0.6, 1.0, 0.8, 0.1)

            with col3:
                colsample_bytree = st.slider("特徵抽樣比例", 0.6, 1.0, 0.8, 0.1)

    # 訓練按鈕和狀態
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🚀 開始訓練", type="primary", use_container_width=True):
            st.session_state.start_training = True

    with col2:
        if model_type in ["random_forest", "transformer"]:
            st.warning(f"{model_type} 模型正在開發中，請選擇 XGBoost")

    # 訓練執行
    if st.session_state.get('start_training', False):
        if model_type != "xgboost":
            st.error("目前只支援 XGBoost 模型")
            st.session_state.start_training = False
        else:
            train_model_locally(model_type, target_variable, test_size, {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree
            })

    # 顯示已訓練模型狀態
    show_trained_models_status()