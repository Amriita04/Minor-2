"""Train regression (plant) and classification (machines); optional XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
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


@dataclass
class TrainedBundle:
    plant_rf: RandomForestRegressor
    plant_xgb: Optional[Any]
    machine_rf: RandomForestClassifier
    machine_lr: LogisticRegression
    plant_metrics: Dict[str, float]
    machine_metrics: Dict[str, Any]
    X_ccpp_test: pd.DataFrame
    y_ccpp_test: np.ndarray
    y_ccpp_pred_rf: np.ndarray
    y_ccpp_pred_xgb: Optional[np.ndarray]
    X_ai_test: pd.DataFrame
    y_ai_test: np.ndarray
    y_ai_proba_rf: np.ndarray
    feature_names_ai: list


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
    plant_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        random_state=random_state,
        n_jobs=-1,
    )
    plant_rf.fit(X_ccpp_tr, y_ccpp_tr)
    y_pred_rf = plant_rf.predict(X_ccpp_te)

    plant_xgb = None
    y_pred_xgb = None
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

    machine_rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    machine_rf.fit(X_ai_tr, y_ai_tr)
    proba_rf = machine_rf.predict_proba(X_ai_te)[:, 1]

    machine_lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    machine_lr.fit(X_ai_tr, y_ai_tr)

    plant_metrics = {
        "r2_rf": float(r2_score(y_ccpp_te, y_pred_rf)),
        "mae_rf_mw": float(mean_absolute_error(y_ccpp_te, y_pred_rf)),
    }
    if y_pred_xgb is not None:
        plant_metrics["r2_xgb"] = float(r2_score(y_ccpp_te, y_pred_xgb))
        plant_metrics["mae_xgb_mw"] = float(mean_absolute_error(y_ccpp_te, y_pred_xgb))

    y_hat_rf = machine_rf.predict(X_ai_te)
    y_hat_lr = machine_lr.predict(X_ai_te)
    machine_metrics = {
        "accuracy_rf": float(accuracy_score(y_ai_te, y_hat_rf)),
        "accuracy_lr": float(accuracy_score(y_ai_te, y_hat_lr)),
        "report_rf": classification_report(y_ai_te, y_hat_rf, zero_division=0),
    }

    return TrainedBundle(
        plant_rf=plant_rf,
        plant_xgb=plant_xgb,
        machine_rf=machine_rf,
        machine_lr=machine_lr,
        plant_metrics=plant_metrics,
        machine_metrics=machine_metrics,
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
    return PlantLstmResult(
        model=model,
        y_test=y_te,
        y_pred=y_pred,
        metrics={
            "r2_lstm": float(r2_score(y_te, y_pred)),
            "mae_lstm_mw": float(mean_absolute_error(y_te, y_pred)),
        },
    )
