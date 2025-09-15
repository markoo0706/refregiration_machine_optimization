"""
緩存與 session_state 工具
"""
import streamlit as st
import time
from typing import Any, Callable, Optional


def get_cached_data(key: str, generate_func: Callable, cache_duration: int = 10) -> Any:
    """
    獲取緩存數據，過期時自動重新生成

    Args:
        key: 緩存鍵值
        generate_func: 數據生成函數
        cache_duration: 緩存持續時間（秒）

    Returns:
        緩存的數據
    """
    cache_key = f"cache_{key}"
    time_key = f"cache_time_{key}"

    current_time = time.time()

    # 檢查是否有緩存且未過期
    if (cache_key in st.session_state and
        time_key in st.session_state and
        current_time - st.session_state[time_key] < cache_duration):
        return st.session_state[cache_key]

    # 生成新數據
    data = generate_func()
    st.session_state[cache_key] = data
    st.session_state[time_key] = current_time

    return data


def clear_cache(pattern: Optional[str] = None):
    """
    清理緩存

    Args:
        pattern: 要清理的緩存模式，None 表示清理所有緩存
    """
    if pattern is None:
        # 清理所有緩存
        cache_keys = [k for k in st.session_state.keys()
                     if k.startswith("cache_")]
    else:
        # 清理匹配模式的緩存
        cache_keys = [k for k in st.session_state.keys()
                     if k.startswith("cache_") and pattern in k]

    for key in cache_keys:
        del st.session_state[key]


def set_session_value(key: str, value: Any):
    """設置 session state 值"""
    st.session_state[key] = value


def get_session_value(key: str, default: Any = None) -> Any:
    """獲取 session state 值"""
    return st.session_state.get(key, default)


def delete_session_value(key: str):
    """刪除 session state 值"""
    if key in st.session_state:
        del st.session_state[key]


def get_cache_info() -> dict:
    """獲取緩存資訊"""
    cache_keys = [k for k in st.session_state.keys() if k.startswith("cache_")]
    time_keys = [k for k in st.session_state.keys() if k.startswith("cache_time_")]

    return {
        "cache_count": len(cache_keys),
        "time_count": len(time_keys),
        "total_keys": len(st.session_state.keys()),
        "cache_keys": cache_keys
    }