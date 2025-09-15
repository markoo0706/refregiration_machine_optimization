"""
API 請求封裝
"""
import requests
import streamlit as st
import os
from typing import Dict, Any, Optional


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                files: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """發送API請求"""
    try:
        url = f"{API_BASE}{endpoint}"

        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            if files:
                response = requests.post(url, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
        else:
            raise ValueError(f"不支援的HTTP方法: {method}")

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API錯誤 {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"連線失敗: {str(e)}")
        return None
    except Exception as e:
        st.error(f"未知錯誤: {str(e)}")
        return None


def check_api_health() -> bool:
    """檢查 API 健康狀態"""
    response = api_request("/health")
    return response is not None