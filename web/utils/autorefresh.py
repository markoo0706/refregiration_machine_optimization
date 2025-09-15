"""
自定義自動刷新功能
使用 Streamlit 官方建議的方法，避免使用 JavaScript 強制刷新
"""
import streamlit as st
import time
from typing import Optional


def st_autorefresh(interval: int = 10, limit: Optional[int] = None,
                   key: str = "autorefresh") -> int:
    """
    自定義自動刷新功能

    Args:
        interval: 刷新間隔（秒）
        limit: 最大刷新次數，None 表示無限制
        key: session state 的鍵值

    Returns:
        當前刷新次數
    """
    # 初始化 session state
    if f"{key}_counter" not in st.session_state:
        st.session_state[f"{key}_counter"] = 0
        st.session_state[f"{key}_last_update"] = time.time()

    # 檢查是否達到限制
    if limit and st.session_state[f"{key}_counter"] >= limit:
        return st.session_state[f"{key}_counter"]

    # 檢查是否需要刷新
    current_time = time.time()
    time_since_last = current_time - st.session_state[f"{key}_last_update"]

    if time_since_last >= interval:
        st.session_state[f"{key}_counter"] += 1
        st.session_state[f"{key}_last_update"] = current_time
        st.rerun()

    return st.session_state[f"{key}_counter"]


def show_refresh_status(interval: int, auto_refresh: bool, key: str = "autorefresh"):
    """
    顯示刷新狀態

    Args:
        interval: 刷新間隔
        auto_refresh: 是否啟用自動刷新
        key: session state 的鍵值
    """
    if auto_refresh:
        refresh_count = st.session_state.get(f"{key}_counter", 0)
        if refresh_count > 0:
            st.caption(f"🔄 自動刷新已啟用 | 間隔: {interval}秒 | 已刷新: {refresh_count}次")
        else:
            st.caption(f"🔄 自動刷新已啟用 | 間隔: {interval}秒")
    else:
        st.caption("⏸️ 自動刷新已關閉")


def reset_refresh_counter(key: str = "autorefresh"):
    """重置刷新計數器"""
    if f"{key}_counter" in st.session_state:
        del st.session_state[f"{key}_counter"]
    if f"{key}_last_update" in st.session_state:
        del st.session_state[f"{key}_last_update"]


def get_refresh_info(key: str = "autorefresh") -> dict:
    """獲取刷新資訊"""
    return {
        "counter": st.session_state.get(f"{key}_counter", 0),
        "last_update": st.session_state.get(f"{key}_last_update", 0),
        "current_time": time.time()
    }