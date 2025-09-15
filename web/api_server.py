import sys
import os
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import requests

# 在容器中確保正確的 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException

# Import the Celery app and tasks
from celery_app import celery_app
from tasks import (
    process_data_task,
    train_model_task,
    optimize_parameters_task
)
from models.schemas import (
    TaskResponse, TaskResult, OptimizationRequest,
    OptimizationResult, ModelMetrics, HealthCheck
)
from constants import ALL_EXTERNAL_VARIABLES, ALL_MONITORING_VARIABLES
from utils.data_generator import generate_mock_realtime_data

app = FastAPI(title="Chiller Optimization API", version="1.0.0")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/upload-data", response_model=TaskResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload data and trigger feature engineering"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    contents = await file.read()
    task = celery_app.send_task('web.tasks.process_data_task', args=[contents, file.filename])
    return {"task_id": task.id, "status": "processing"}


@app.post("/train/temperature", response_model=TaskResponse)
async def train_temperature_model():
    """Train temperature prediction model"""
    task = celery_app.send_task('web.tasks.train_model_task', kwargs={'model_name': 'xgboost', 'target_column': 'cooling_tower_outlet_temp_c'})
    return {"task_id": task.id, "status": "training"}


@app.post("/train/power", response_model=TaskResponse)
async def train_power_model():
    """Train power consumption prediction model"""
    task = celery_app.send_task('web.tasks.train_model_task', kwargs={'model_name': 'xgboost', 'target_column': 'cooling_system_total_power_kw'})
    return {"task_id": task.id, "status": "training"}


@app.post("/train/cop", response_model=TaskResponse)
async def train_cop_model():
    """Train COP prediction model"""
    task = celery_app.send_task('web.tasks.train_model_task', kwargs={'model_name': 'xgboost', 'target_column': 'cooling_system_cop'})
    return {"task_id": task.id, "status": "training"}


@app.post("/optimize", response_model=TaskResponse)
async def optimize_parameters(request: OptimizationRequest):
    """Generate optimal chiller parameters"""
    task = celery_app.send_task('web.tasks.optimize_parameters_task', args=[request.model_dump()])
    return {"task_id": task.id, "status": "optimizing"}


@app.get("/logs/{task_id}", response_model=TaskResult)
async def get_task_logs(task_id: str):
    """Get task execution logs"""
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    # Handle different result states properly
    if result.ready():
        if result.successful():
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if isinstance(result.result, dict) else {"data": result.result},
                "info": "Task completed successfully"
            }
        else:
            # Task failed
            return {
                "task_id": task_id,
                "status": result.status,
                "result": {"error": str(result.result)} if result.result else {"error": "Unknown error"},
                "info": str(result.info) if result.info is not None else "Task failed"
            }
    else:
        # Task is still pending or in progress
        return {
            "task_id": task_id,
            "status": result.status,
            "result": None,
            "info": str(result.info) if result.info is not None else f"Task is {result.status.lower()}"
        }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/current-system-state")
async def get_current_system_state():
    """獲取當前系統狀態 - 包含完整的監控和環境變數供最佳化使用"""
    try:
        # 構建當前系統狀態，確保包含所有簡化特徵工程所需的變數
        current_state = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "external_variables": {},
            "monitoring_variables": {},
            "control_variables": {},
            "derived_features": {},
            "system_metrics": {}
        }
        
        # === 外部環境變數（無法控制的環境因子）===
        current_state["external_variables"] = {
            "ambient_temperature_c": 25.0 + np.random.normal(0, 3),  # 22-28°C
            "ambient_humidity_rh": 65.0 + np.random.normal(0, 10),   # 55-75%
            # 製程需求（來自反應釜）
            "tic206a_reactor_temp_c": 80.0 + np.random.normal(0, 5),
            "tic206b_reactor_temp_c": 82.0 + np.random.normal(0, 5),
            "tic206c_reactor_temp_c": 81.0 + np.random.normal(0, 5),
            "tic206d_reactor_temp_c": 79.0 + np.random.normal(0, 5),
            "tic607a_reactor_oil_temp_c": 70.0 + np.random.normal(0, 3),
            "tic607b_reactor_oil_temp_c": 71.0 + np.random.normal(0, 3),
            "tic607c_reactor_oil_temp_c": 69.0 + np.random.normal(0, 3),
            "tic607d_reactor_oil_temp_c": 72.0 + np.random.normal(0, 3),
        }
        
        # === 監控變數（系統狀態指標）===
        current_state["monitoring_variables"] = {
            # 溫度監控
            "cooling_tower_outlet_temp_c": 24.0 + np.random.normal(0, 2),
            "cooling_tower_return_temp_c": 29.0 + np.random.normal(0, 2),
            "rf501a_chilled_inlet_temp_c": 12.0 + np.random.normal(0, 1),
            "rf501a_chilled_outlet_temp_c": 7.0 + np.random.normal(0, 0.5),
            "rf501a_cooling_inlet_temp_c": 29.0 + np.random.normal(0, 1),
            "rf501a_cooling_outlet_temp_c": 32.0 + np.random.normal(0, 1),
            "rf501b_chilled_inlet_temp_c": 12.2 + np.random.normal(0, 1),
            "rf501b_chilled_outlet_temp_c": 7.2 + np.random.normal(0, 0.5),
            "rf501b_cooling_inlet_temp_c": 29.2 + np.random.normal(0, 1),
            "rf501b_cooling_outlet_temp_c": 32.2 + np.random.normal(0, 1),
            "rf501c_chilled_inlet_temp_c": 11.8 + np.random.normal(0, 1),
            "rf501c_chilled_outlet_temp_c": 6.8 + np.random.normal(0, 0.5),
            "rf501c_cooling_inlet_temp_c": 28.8 + np.random.normal(0, 1),
            "rf501c_cooling_outlet_temp_c": 31.8 + np.random.normal(0, 1),
            "chilled_water_return_temp_c": 12.0 + np.random.normal(0, 1),
            "chilled_water_tank_temp_c": 6.5 + np.random.normal(0, 0.5),
            
            # 流量監控
            "rf501a_chilled_flow_m3_hr": 150.0 + np.random.normal(0, 15),
            "rf501b_chilled_flow_m3_hr": 148.0 + np.random.normal(0, 15),
            "rf501c_chilled_flow_m3_hr": 152.0 + np.random.normal(0, 15),
            "e401a_cooling_flow_m3_hr": 200.0 + np.random.normal(0, 20),
            "e401b_cooling_flow_m3_hr": 195.0 + np.random.normal(0, 20),
            "e401c_cooling_flow_m3_hr": 205.0 + np.random.normal(0, 20),
            "e401d_cooling_flow_m3_hr": 198.0 + np.random.normal(0, 20),
            
            # 電流監控
            "rf501a_current_a": 180.0 + np.random.normal(0, 20),
            "rf501b_current_a": 175.0 + np.random.normal(0, 20),
            "rf501c_current_a": 185.0 + np.random.normal(0, 20),
            "fan_510a_current_a": 35.0 + np.random.normal(0, 5),
            "fan_510b_current_a": 33.0 + np.random.normal(0, 5),
            "fan_510c_current_a": 37.0 + np.random.normal(0, 5),
            
            # 壓力監控
            "cooling_water_outlet_pressure_kg_cm2": 2.5 + np.random.normal(0, 0.2),
            "chilled_water_pressure_diff_kg_cm2": 1.2 + np.random.normal(0, 0.1),
        }
        
        # === 控制變數的當前狀態（可被最佳化算法調整）===
        current_state["control_variables"] = {
            # 冷卻塔控制
            "cooling_tower_opening_pct": 45.0 + np.random.normal(0, 5),
            
            # 風扇功率控制
            "fan_510a_power_kw": 80.0 + np.random.normal(0, 10),
            "fan_510b_power_kw": 75.0 + np.random.normal(0, 10),
            "fan_510c_power_kw": 85.0 + np.random.normal(0, 10),
            
            # 泵浦功率控制
            "cooling_pump_g511a_power_kw": 25.0 + np.random.normal(0, 3),
            "cooling_pump_g511b_power_kw": 24.0 + np.random.normal(0, 3),
            "cooling_pump_g511c_power_kw": 26.0 + np.random.normal(0, 3),
            "cooling_pump_g511x_power_kw": 23.0 + np.random.normal(0, 3),
            "cooling_pump_g511y_power_kw": 27.0 + np.random.normal(0, 3),
            "chilled_pump_g501a_power_kw": 30.0 + np.random.normal(0, 4),
            "chilled_pump_g501b_power_kw": 28.0 + np.random.normal(0, 4),
            "chilled_pump_g501c_power_kw": 32.0 + np.random.normal(0, 4),
            
            # 冷卻器溫度控制
            "e401a_cooling_outlet_temp_c": 32.0 + np.random.normal(0, 1),
            "e401b_cooling_outlet_temp_c": 31.5 + np.random.normal(0, 1),
            "e401c_cooling_outlet_temp_c": 32.5 + np.random.normal(0, 1),
            "e401d_cooling_outlet_temp_c": 31.8 + np.random.normal(0, 1),
            
            # 閥門開度控制
            "tic607a_cooling_valve_opening_pct": 55.0 + np.random.normal(0, 5),
            "tic607b_cooling_valve_opening_pct": 58.0 + np.random.normal(0, 5),
            "tic607c_cooling_valve_opening_pct": 52.0 + np.random.normal(0, 5),
            "tic607d_cooling_valve_opening_pct": 60.0 + np.random.normal(0, 5),
        }
        
        # === 計算衍生特徵（基於當前控制變數）===
        # 總功率
        total_fan_power = (current_state["control_variables"]["fan_510a_power_kw"] + 
                          current_state["control_variables"]["fan_510b_power_kw"] + 
                          current_state["control_variables"]["fan_510c_power_kw"])
        
        total_pump_power = sum([
            current_state["control_variables"]["cooling_pump_g511a_power_kw"],
            current_state["control_variables"]["cooling_pump_g511b_power_kw"],
            current_state["control_variables"]["cooling_pump_g511c_power_kw"],
            current_state["control_variables"]["cooling_pump_g511x_power_kw"],
            current_state["control_variables"]["cooling_pump_g511y_power_kw"],
            current_state["control_variables"]["chilled_pump_g501a_power_kw"],
            current_state["control_variables"]["chilled_pump_g501b_power_kw"],
            current_state["control_variables"]["chilled_pump_g501c_power_kw"],
        ])
        
        current_state["derived_features"] = {
            "total_fan_power_kw": total_fan_power,
            "total_pump_power_kw": total_pump_power,
            "cooling_tower_temp_diff": (current_state["monitoring_variables"]["cooling_tower_return_temp_c"] - 
                                       current_state["monitoring_variables"]["cooling_tower_outlet_temp_c"]),
            "ambient_fan_interaction": (current_state["external_variables"]["ambient_temperature_c"] * total_fan_power),
            "active_fans_count": sum([1 for power in [
                current_state["control_variables"]["fan_510a_power_kw"],
                current_state["control_variables"]["fan_510b_power_kw"], 
                current_state["control_variables"]["fan_510c_power_kw"]
            ] if power > 5]),  # 功率>5kW視為運行
            
            # 非線性特徵（特徵工程需要的）
            "cooling_tower_opening_pct_squared": (current_state["control_variables"]["cooling_tower_opening_pct"] / 100.0) ** 2,
            "fan_510a_power_kw_sqrt": np.sqrt(max(current_state["control_variables"]["fan_510a_power_kw"], 0)),
            "fan_510b_power_kw_sqrt": np.sqrt(max(current_state["control_variables"]["fan_510b_power_kw"], 0)),
            "fan_510c_power_kw_sqrt": np.sqrt(max(current_state["control_variables"]["fan_510c_power_kw"], 0)),
        }
        
        # === 系統指標 ===
        current_state["system_metrics"] = {
            "total_power_estimate": total_fan_power + total_pump_power,
            "average_fan_current": np.mean([
                current_state["monitoring_variables"]["fan_510a_current_a"],
                current_state["monitoring_variables"]["fan_510b_current_a"],
                current_state["monitoring_variables"]["fan_510c_current_a"]
            ]),
            "cooling_system_total_power_kw": total_fan_power + total_pump_power * 0.8,  # 簡化計算
            "cooling_system_cop": max(1.0, 300.0 / max(total_fan_power + total_pump_power * 0.8, 50)),  # 簡化COP
        }
        
        return current_state
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system state: {str(e)}")