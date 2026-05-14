"""Optional LSTM regressor on pseudo-sequenced CCPP windows (TensorFlow if installed)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from industrial_energy_optimizer.preprocess import prepare_ccpp_sequence_arrays


def train_lstm_ccpp_demo(
    df,
    seq_len: int = 24,
    epochs: int = 12,
    batch_size: int = 256,
    random_state: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with keys mae, r2, y_pred_te (1d) or None if TensorFlow missing.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception:
        return None

    X_tr, X_te, y_tr, y_te = prepare_ccpp_sequence_arrays(df, seq_len=seq_len)
    keras.utils.set_random_seed(random_state)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_tr.shape[1], X_tr.shape[2])),
            keras.layers.LSTM(48, dropout=0.1),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    pred = model.predict(X_te, verbose=0).ravel()
    mae = float(np.mean(np.abs(pred - y_te)))
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"mae_mw": mae, "r2": r2, "y_pred_te": pred, "y_te": y_te}
