from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: Optional[str] = Field(None, description="Current API version")


class HealthCheck(BaseModel):
    status: str = Field(..., description="Service status")


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")


class TaskResult(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    info: Optional[str] = Field(None, description="Task information")


class ModelMetrics(BaseModel):
    r2: float = Field(..., description="R-squared score")
    rmse: float = Field(..., description="Root mean squared error")
    mae: float = Field(..., description="Mean absolute error")
    mape: float = Field(..., description="Mean absolute percentage error")


class OptimizationResult(BaseModel):
    solutions: List[Dict[str, Any]] = Field(..., description="Pareto optimal solutions")
    metrics: Dict[str, float] = Field(..., description="Optimization metrics")


class OptimizationRequest(BaseModel):
    objective: str = Field(..., description="Primary optimization objective")
    target_temp: float = Field(..., description="Target temperature")
    algorithm: str = Field("nsga2", description="Algorithm identifier")
    max_iterations: int = Field(100, ge=10, le=500)
    population_size: int = Field(100, ge=10, le=500)
    weight_power: float = Field(1.0, ge=0.0, le=2.0)
    weight_efficiency: float = Field(0.8, ge=0.0, le=2.0)


class OptimizationResultItem(BaseModel):
    params: Dict[str, float]
    objectives: Dict[str, float]
    feasible: bool = True


class OptimizationResponse(BaseModel):
    items: List[OptimizationResultItem]
    algorithm: str
    objective: str
    generations: int


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: str = Field("xgboost", description="Model identifier")
    target: str = Field(..., description="Target variable name")
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    params: Optional[Dict[str, Any]] = None


class TrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: str
    target: str
    metrics: Dict[str, float]
    model_path: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


__all__ = [
    "HealthResponse",
    "HealthCheck",
    "TaskResponse",
    "TaskResult",
    "ModelMetrics",
    "OptimizationResult",
    "OptimizationRequest",
    "OptimizationResultItem",
    "OptimizationResponse",
    "TrainRequest",
    "TrainResponse",
    "TaskStatusResponse",
]
