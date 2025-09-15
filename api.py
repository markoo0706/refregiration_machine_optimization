"""
FastAPI 後端服務器啟動器

使用方式：
    python api.py
    或
    uvicorn web.api_server:app --reload
"""

import sys
import os

# 添加當前目錄到 Python path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.api_server:app", host="0.0.0.0", port=8000, reload=True)