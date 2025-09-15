"""
預測模型管理系統
===========================

包含多種機器學習模型用於冷卻塔系統預測：
1. XGBoost（主要模型）
2. Random Forest（預留）
3. Transformer（預留）

設計原則：
- 統一的模型介面
- 支援時間序列特徵（lagging features）
- 模型表現評估與比較
- 易於擴展新模型

"""

import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 模型依賴
import xgboost as xgb

# 抑制警告
warnings.filterwarnings('ignore', category=UserWarning)

from constants import DATA_DIR, MODELS_DIR, TIME_COLUMN
from feature_engineering import FeatureProcessor


class BaseModel(ABC):
    """基礎模型抽象類別"""

    def __init__(self, model_name: str, model_type: str = "regression"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_names = []
        self.is_trained = False
        self.training_metrics = {}

        # 確保模型目錄存在
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")

    @abstractmethod
    def _create_model(self, **kwargs):
        """創建模型實例"""
        pass

    @abstractmethod
    def _train_model(self, X_train, y_train, X_val=None, y_val=None):
        """訓練模型"""
        pass

    def train(self, X: pd.DataFrame, y: pd.DataFrame,
              target_columns: Optional[List[str]] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              **model_kwargs) -> Dict[str, Any]:
        """
        訓練模型

        參數:
        - X: 特徵數據
        - y: 目標數據
        - target_columns: 要預測的目標欄位
        - test_size: 測試集比例
        - random_state: 隨機種子
        - model_kwargs: 模型特定參數

        回傳:
        - 訓練指標字典
        """

        # 準備特徵和目標
        feature_cols = [col for col in X.columns if col != TIME_COLUMN]
        self.feature_names = feature_cols

        if target_columns is None:
            target_columns = [col for col in y.columns if col != TIME_COLUMN]
        self.target_names = target_columns

        # 提取數據
        X_data = X[feature_cols].values
        y_data = y[target_columns].values

        # 多目標預測時使用第一個目標（可擴展為多輸出）
        if len(target_columns) > 1:
            print(f"多目標預測，使用第一個目標: {target_columns[0]}")
            y_data = y_data[:, 0]
            self.target_names = [target_columns[0]]

        # 分割訓練/測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state
        )

        # 特徵標準化（僅針對某些模型）
        if self.model_name in ['random_forest', 'transformer']:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # 創建和訓練模型
        self.model = self._create_model(**model_kwargs)
        self._train_model(X_train, y_train, X_test, y_test)

        # 評估模型
        y_pred = self.model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, len(X_train))

        self.training_metrics = metrics
        self.is_trained = True

        # 保存模型
        self.save_model()

        return metrics

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """預測"""
        if not self.is_trained or self.model is None:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")

        if isinstance(X, pd.DataFrame):
            X_data = X[self.feature_names].values
        else:
            X_data = X

        # 標準化（如需要）
        if self.model_name in ['random_forest', 'transformer']:
            X_data = self.scaler.transform(X_data)

        return self.model.predict(X_data)

    def _calculate_metrics(self, y_true, y_pred, n_train_samples) -> Dict[str, float]:
        """計算模型評估指標"""
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100),
            'samples': int(n_train_samples),
            'test_samples': int(len(y_true))
        }

    def save_model(self):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'training_metrics': self.training_metrics,
            'model_name': self.model_name
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """載入模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """取得特徵重要性"""
        if not self.is_trained:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'get_score'):  # XGBoost
            importance_dict = self.model.get_score(importance_type='weight')
            # 轉換特徵名稱格式
            return {f"f{i}": importance_dict.get(f"f{i}", 0.0)
                   for i in range(len(self.feature_names))}
        else:
            return None


class XGBoostModel(BaseModel):
    """XGBoost 預測模型 - 主要使用模型"""

    def __init__(self):
        super().__init__("xgboost", "regression")

    def _create_model(self, **kwargs):
        """創建 XGBoost 模型"""
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)

        return xgb.XGBRegressor(**default_params)

    def _train_model(self, X_train, y_train, X_val=None, y_val=None):
        """訓練 XGBoost 模型"""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """取得 XGBoost 特徵重要性"""
        if not self.is_trained:
            return None

        try:
            # XGBoost 有多種特徵重要性類型
            importance_weight = self.model.get_booster().get_score(importance_type='weight')
            importance_gain = self.model.get_booster().get_score(importance_type='gain')

            # 轉換為實際特徵名稱
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_key = f"f{i}"
                feature_importance[feature_name] = {
                    'weight': importance_weight.get(feature_key, 0.0),
                    'gain': importance_gain.get(feature_key, 0.0)
                }

            return feature_importance
        except:
            # Fallback 到 sklearn 風格的特徵重要性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names, importances))
            return None


class RandomForestModel(BaseModel):
    """Random Forest 模型 - 預留擴展"""

    def __init__(self):
        super().__init__("random_forest", "regression")

    def _create_model(self, **kwargs):
        """創建 Random Forest 模型 - 預留"""
        # TODO: 實現 Random Forest 模型
        raise NotImplementedError("Random Forest 模型待實現")

    def _train_model(self, X_train, y_train, X_val=None, y_val=None):
        """訓練 Random Forest 模型 - 預留"""
        # TODO: 實現訓練邏輯
        raise NotImplementedError("Random Forest 訓練邏輯待實現")


class TransformerModel(BaseModel):
    """Transformer 時間序列模型 - 預留擴展"""

    def __init__(self):
        super().__init__("transformer", "time_series")

    def _create_model(self, **kwargs):
        """創建 Transformer 模型 - 預留"""
        # TODO: 實現 Transformer 模型
        raise NotImplementedError("Transformer 模型待實現")

    def _train_model(self, X_train, y_train, X_val=None, y_val=None):
        """訓練 Transformer 模型 - 預留"""
        # TODO: 實現訓練邏輯
        raise NotImplementedError("Transformer 訓練邏輯待實現")


class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.available_models = {
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel,
            'transformer': TransformerModel
        }
        self.trained_models = {}

    def create_model(self, model_name: str) -> BaseModel:
        """創建模型實例"""
        if model_name not in self.available_models:
            raise ValueError(f"不支援的模型: {model_name}")

        return self.available_models[model_name]()

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.DataFrame,
                   **kwargs) -> Dict[str, Any]:
        """訓練指定模型"""
        model = self.create_model(model_name)
        metrics = model.train(X, y, **kwargs)
        self.trained_models[model_name] = model
        return metrics

    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """取得已訓練的模型"""
        return self.trained_models.get(model_name)

    def load_model(self, model_name: str) -> BaseModel:
        """載入已保存的模型"""
        model = self.create_model(model_name)
        model.load_model()
        self.trained_models[model_name] = model
        return model

    def compare_models(self, X: pd.DataFrame, y: pd.DataFrame,
                      models: List[str] = ['xgboost']) -> pd.DataFrame:
        """比較多個模型的表現"""
        results = []

        for model_name in models:
            if model_name not in self.available_models:
                continue

            try:
                print(f"正在訓練 {model_name}...")
                metrics = self.train_model(model_name, X, y)
                metrics['model'] = model_name
                results.append(metrics)
            except NotImplementedError:
                print(f"{model_name} 模型尚未實現")
                continue
            except Exception as e:
                print(f"{model_name} 訓練失敗: {str(e)}")
                continue

        return pd.DataFrame(results)


# 便利函數
def train_cooling_system_model(data_path: Optional[str] = None,
                             model_name: str = 'xgboost',
                             target_columns: Optional[List[str]] = None) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    訓練冷卻系統預測模型的便利函數

    參數:
    - data_path: 數據文件路徑
    - model_name: 模型名稱
    - target_columns: 目標欄位

    回傳:
    - (模型實例, 訓練指標)
    """

    # 數據處理
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "processed.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"數據文件不存在: {data_path}")

    # 特徵工程
    processor = FeatureProcessor(verbose=True)
    df_raw = pd.read_csv(data_path)
    df_processed = processor.process(df_raw)

    # 準備訓練數據
    X, y = processor.get_training_data(df_processed, target_columns)

    # 訓練模型
    manager = ModelManager()
    metrics = manager.train_model(model_name, X, y)
    model = manager.get_model(model_name)

    return model, metrics


__all__ = [
    'BaseModel', 'XGBoostModel', 'RandomForestModel', 'TransformerModel',
    'ModelManager', 'train_cooling_system_model'
]