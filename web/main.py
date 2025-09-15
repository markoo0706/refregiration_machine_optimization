"""
冰水主機最佳化監控系統 - Streamlit 主入口

重構後的模組化版本
使用 Streamlit 官方建議的自動刷新方法，取代 JavaScript 強制刷新

功能：
1. 數據上傳與處理
2. 模型訓練（含增強視覺化和自動保存）
3. 多目標最佳化
4. 系統監控
5. 即時監控（優化的自動刷新）
"""

import streamlit as st

# 設定頁面配置
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="冰水主機最佳化系統",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.page_config_set = True

# 導入各個頁面模組
from web.views.realtime import show_realtime_dashboard
from web.views.upload import show_data_upload
from web.views.training import show_model_training
from web.views.optimization import show_optimization
from web.views.monitoring import show_monitoring
from web.utils.api import api_request


def main():
    """主應用程式"""
    st.title("冰水主機最佳化系統 (v2.0.0)")
    st.markdown("---")

    # 主選單 - 使用新的模組化頁面
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "即時監控", "數據上傳", "模型訓練", "多目標最佳化", "系統監控"
    ])

    with tab1:
        show_realtime_dashboard()

    with tab2:
        show_data_upload()

    with tab3:
        show_model_training()

    with tab4:
        show_optimization()

    with tab5:
        show_monitoring()

    # 頁腳資訊
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🔄 **自動刷新**: 使用 Streamlit 原生方法")

    with col2:
        st.caption("📊 **視覺化**: 增強的模型表現圖表")

    with col3:
        st.caption("💾 **模型保存**: 自動時間戳命名")


if __name__ == "__main__":
    main()