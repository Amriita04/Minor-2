"""Train regression (plant) and classification (machines); optional XGBoost.

Comprehensive evaluation with:
- Regression: R², MAE, RMSE, MAPE
- Classification: accuracy, precision, recall, F1, confusion matrix
- Model comparison dashboard-ready metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

from industrial_energy_optimizer.preprocess import prepare_ccpp_sequence_arrays

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TF = True
except Exception:
    HAS_TF = False


def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (% units)."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


@dataclass
class RegressionMetrics:
    """Container for regression model evaluation metrics."""
    r2: float
    mae: float
    rmse: float
    mape: float
    
    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary for dashboard display."""
        return {
            "r2": round(self.r2, 4),
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "mape": round(self.mape, 2),
        }


@dataclass
class ClassificationMetrics:
    """Container for classification model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    classification_report: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for dashboard display."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


@dataclass
class TrainedBundle:
    """Complete trained model bundle with metrics and predictions."""
    plant_rf: RandomForestRegressor
    plant_xgb: Optional[Any]
    machine_rf: RandomForestClassifier
    machine_lr: LogisticRegression
    plant_metrics_rf: RegressionMetrics
    plant_metrics_xgb: Optional[RegressionMetrics]
    machine_metrics_rf: ClassificationMetrics
    machine_metrics_lr: ClassificationMetrics
    X_ccpp_test: pd.DataFrame
    y_ccpp_test: np.ndarray
    y_ccpp_pred_rf: np.ndarray
    y_ccpp_pred_xgb: Optional[np.ndarray]
    X_ai_test: pd.DataFrame
    y_ai_test: np.ndarray
    y_ai_proba_rf: np.ndarray
    feature_names_ai: list
    # Legacy fields for backward compatibility with existing code
    plant_metrics: Dict[str, float] = field(init=False)
    machine_metrics: Dict[str, Any] = field(init=False)
    
    def __post_init__(self):
        """Build legacy metric dicts for backward compatibility."""
        # Plant metrics dict
        self.plant_metrics = {
            "r2_rf": self.plant_metrics_rf.r2,
            "mae_rf_mw": self.plant_metrics_rf.mae,
            "rmse_rf_mw": self.plant_metrics_rf.rmse,
            "mape_rf": self.plant_metrics_rf.mape,
        }
        if self.plant_metrics_xgb:
            self.plant_metrics.update({
                "r2_xgb": self.plant_metrics_xgb.r2,
                "mae_xgb_mw": self.plant_metrics_xgb.mae,
                "rmse_xgb_mw": self.plant_metrics_xgb.rmse,
                "mape_xgb": self.plant_metrics_xgb.mape,
            })
        
        # Machine metrics dict
        self.machine_metrics = {
            "accuracy_rf": self.machine_metrics_rf.accuracy,
            "precision_rf": self.machine_metrics_rf.precision,
            "recall_rf": self.machine_metrics_rf.recall,
            "f1_rf": self.machine_metrics_rf.f1,
            "confusion_matrix_rf": self.machine_metrics_rf.confusion_matrix.tolist(),
            "report_rf": self.machine_metrics_rf.classification_report,
            "accuracy_lr": self.machine_metrics_lr.accuracy,
            "precision_lr": self.machine_metrics_lr.precision,
            "recall_lr": self.machine_metrics_lr.recall,
            "f1_lr": self.machine_metrics_lr.f1,
            "confusion_matrix_lr": self.machine_metrics_lr.confusion_matrix.tolist(),
            "report_lr": self.machine_metrics_lr.classification_report,
        }


def _evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> RegressionMetrics:
    """Evaluate regression predictions with comprehensive metrics."""
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=_calculate_rmse(y_true, y_pred),
        mape=_calculate_mape(y_true, y_pred),
    )


def _evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> ClassificationMetrics:
    """Evaluate classification predictions with comprehensive metrics."""
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0, average="weighted")),
        recall=float(recall_score(y_true, y_pred, zero_division=0, average="weighted")),
        f1=float(f1_score(y_true, y_pred, zero_division=0, average="weighted")),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        classification_report=classification_report(y_true, y_pred, zero_division=0),
    )


def train_models(
    X_ccpp_tr: pd.DataFrame,
    X_ccpp_te: pd.DataFrame,
    y_ccpp_tr: np.ndarray,
    y_ccpp_te: np.ndarray,
    X_ai_tr: pd.DataFrame,
    X_ai_te: pd.DataFrame,
    y_ai_tr: np.ndarray,
    y_ai_te: np.ndarray,
    random_state: int = 42,
) -> TrainedBundle:
    """Train all regression and classification models with comprehensive evaluation."""
    
    # ========== PLANT REGRESSION MODELS ==========
    plant_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        random_state=random_state,
        n_jobs=-1,
    )
    plant_rf.fit(X_ccpp_tr, y_ccpp_tr)
    y_pred_rf = plant_rf.predict(X_ccpp_te)
    metrics_rf = _evaluate_regression(y_ccpp_te, y_pred_rf)

    plant_xgb = None
    y_pred_xgb = None
    metrics_xgb = None
    
    if HAS_XGB:
        plant_xgb = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
        plant_xgb.fit(X_ccpp_tr, y_ccpp_tr)
        y_pred_xgb = plant_xgb.predict(X_ccpp_te)
        metrics_xgb = _evaluate_regression(y_ccpp_te, y_pred_xgb)

    # ========== MACHINE CLASSIFICATION MODELS ==========
    machine_rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    machine_rf.fit(X_ai_tr, y_ai_tr)
    y_pred_rf_clf = machine_rf.predict(X_ai_te)
    proba_rf = machine_rf.predict_proba(X_ai_te)[:, 1]
    metrics_rf_clf = _evaluate_classification(y_ai_te, y_pred_rf_clf)

    machine_lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    machine_lr.fit(X_ai_tr, y_ai_tr)
    y_pred_lr = machine_lr.predict(X_ai_te)
    metrics_lr = _evaluate_classification(y_ai_te, y_pred_lr)

    return TrainedBundle(
        plant_rf=plant_rf,
        plant_xgb=plant_xgb,
        machine_rf=machine_rf,
        machine_lr=machine_lr,
        plant_metrics_rf=metrics_rf,
        plant_metrics_xgb=metrics_xgb,
        machine_metrics_rf=metrics_rf_clf,
        machine_metrics_lr=metrics_lr,
        X_ccpp_test=X_ccpp_te,
        y_ccpp_test=y_ccpp_te,
        y_ccpp_pred_rf=y_pred_rf,
        y_ccpp_pred_xgb=y_pred_xgb,
        X_ai_test=X_ai_te,
        y_ai_test=y_ai_te,
        y_ai_proba_rf=proba_rf,
        feature_names_ai=list(X_ai_te.columns),
    )


def build_forecast_curve(y_pred: np.ndarray, sort: bool = True) -> pd.DataFrame:
    """Dashboard-friendly 'demand / output scenario' curve from model predictions."""
    s = np.sort(y_pred) if sort else y_pred.copy()
    return pd.DataFrame({"scenario_index": np.arange(len(s)), "forecast_mw": s})


@dataclass
class PlantLstmResult:
    model: Optional[Any]
    y_test: np.ndarray
    y_pred: Optional[np.ndarray]
    metrics: Dict[str, float]


def train_plant_lstm(
    df_ccpp: pd.DataFrame,
    seq_len: int = 24,
    epochs: int = 12,
    random_state: int = 42,
) -> PlantLstmResult:
    """Sequence regression on pseudo-time windows (LSTM demo)."""
    if not HAS_TF:
        return PlantLstmResult(
            model=None,
            y_test=np.array([]),
            y_pred=None,
            metrics={"note": "TensorFlow not installed; skipped LSTM."},
        )
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)
    X_tr, X_te, y_tr, y_te = prepare_ccpp_sequence_arrays(df_ccpp, seq_len=seq_len)
    if len(X_tr) < 100 or len(X_te) < 20:
        return PlantLstmResult(
            model=None,
            y_test=y_te,
            y_pred=None,
            metrics={"note": "Insufficient sequence samples."},
        )
    n_feat = int(X_tr.shape[2])
    inp = keras.Input(shape=(seq_len, n_feat))
    x = layers.LSTM(32)(inp)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_tr,
        y_tr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=256,
        verbose=0,
    )
    y_pred = model.predict(X_te, verbose=0).astype(float).ravel()
    lstm_metrics = _evaluate_regression(y_te, y_pred)
    return PlantLstmResult(
        model=model,
        y_test=y_te,
        y_pred=y_pred,
        metrics={
            "r2_lstm": lstm_metrics.r2,
            "mae_lstm_mw": lstm_metrics.mae,
            "rmse_lstm_mw": lstm_metrics.rmse,
            "mape_lstm": lstm_metrics.mape,
        },
    )
