"""
數據上傳頁面
"""
import streamlit as st
import time
from utils.api import api_request


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


def show_data_upload():
    """數據上傳頁面"""
    st.header("數據上傳與處理")

    uploaded_file = st.file_uploader("選擇CSV文件", type="csv")

    if uploaded_file and st.button("上傳並處理數據"):
        files = {"file": uploaded_file}
        response = api_request("/upload-data", method="POST", files=files)

        if response:
            st.success(f"上傳成功，任務ID: {response.get('task_id')}")

            # 輪詢任務狀態
            task_id = response.get('task_id')
            if task_id:
                check_task_status(task_id)