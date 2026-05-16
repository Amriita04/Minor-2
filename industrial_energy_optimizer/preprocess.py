"""Preprocessing: cleaning, normalization, machine-wise consumption aggregates."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CCPP_FEATURE_COLS = ["AT", "V", "AP", "RH"]
CCPP_TARGET = "PE"

AI4I_SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def _clean_ccpp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CCPP_FEATURE_COLS + [CCPP_TARGET]:
        if c not in out.columns:
            raise ValueError(f"CCPP missing column {c!r}; got {list(out.columns)}")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _clean_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in AI4I_SENSOR_COLS + ["Machine failure"] + AI4I_FAILURE_COLS:
        if c not in out.columns:
            raise ValueError(f"AI4I missing column {c!r}; got {list(out.columns)}")
    out["any_failure_mode"] = out[AI4I_FAILURE_COLS].max(axis=1).astype(int)
    out["failure_or_mode"] = (
        (out["Machine failure"].astype(int) | out["any_failure_mode"]).astype(int)
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def mechanical_power_kw(torque_nm: pd.Series, rpm: pd.Series) -> pd.Series:
    """Approx mechanical power (kW) from torque and speed (sign-agnostic demo)."""
    omega = rpm * (2.0 * np.pi / 60.0)
    return (torque_nm * omega) / 1000.0


def add_idle_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic idle / unloaded running: high speed with low torque vs cohort."""
    rpm = df["Rotational speed [rpm]"].astype(float)
    tq = df["Torque [Nm]"].astype(float)
    rpm_z = (rpm - rpm.median()) / (rpm.std() + 1e-9)
    tq_z = (tq - tq.median()) / (tq.std() + 1e-9)
    df = df.copy()
    df["idle_like"] = ((rpm_z > 0.5) & (tq_z < -0.5)).astype(int)
    df["power_kw_proxy"] = mechanical_power_kw(tq, rpm)
    return df


def add_machine_fleet_id(df: pd.DataFrame, n_machines: int = 50) -> pd.DataFrame:
    """Map each row to a synthetic fleet machine (UDI is unique per row in AI4I)."""
    df = df.copy()
    if "UDI" not in df.columns:
        raise ValueError("AI4I expects column 'UDI' (UCI export).")
    size = max(1, len(df) // n_machines)
    df["machine_id"] = ((df["UDI"].astype(int) - 1) // size).astype(int) + 1
    return df


def add_machine_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic machine type based on failure patterns (M1-M5)."""
    df = df.copy()
    if "failure_or_mode" not in df.columns:
        raise ValueError("add_machine_type requires 'failure_or_mode' column")
    if "machine_id" not in df.columns:
        raise ValueError("add_machine_type requires 'machine_id' column")
    
    # Create deterministic machine types based on machine_id
    df["Type"] = "M" + ((df["machine_id"] - 1) % 5 + 1).astype(str)
    return df


def build_machine_wise_table(df_ai: pd.DataFrame) -> pd.DataFrame:
    """Per-machine fleet unit: consumption-style aggregates for dashboard micro view."""
    need = {"machine_id", "power_kw_proxy", "failure_or_mode", "idle_like", "Type"}
    missing = need - set(df_ai.columns)
    if missing:
        raise ValueError(f"build_machine_wise_table missing columns: {sorted(missing)}")
    gcols = ["machine_id", "Type"]
    count_col = "UDI" if "UDI" in df_ai.columns else "machine_id"
    agg = (
        df_ai.groupby(gcols, as_index=False)
        .agg(
            records=(count_col, "count"),
            mean_air_K=("Air temperature [K]", "mean"),
            mean_process_K=("Process temperature [K]", "mean"),
            mean_rpm=("Rotational speed [rpm]", "mean"),
            mean_torque=("Torque [Nm]", "mean"),
            mean_tool_wear=("Tool wear [min]", "mean"),
            mean_power_kw=("power_kw_proxy", "mean"),
            total_energy_kwh_est=("power_kw_proxy", "sum"),
            failures=("failure_or_mode", "sum"),
            idle_hours=("idle_like", "sum"),
        )
    )
    return agg


def prepare_ccpp_splits(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    df = _clean_ccpp(df)
    X = df[CCPP_FEATURE_COLS].values
    y = df[CCPP_TARGET].values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return (
        pd.DataFrame(X_tr_s, columns=CCPP_FEATURE_COLS),
        pd.DataFrame(X_te_s, columns=CCPP_FEATURE_COLS),
        y_tr,
        y_te,
        scaler,
    )


def preprocess_ai4i_full(df: pd.DataFrame) -> pd.DataFrame:
    """Complete AI4I preprocessing pipeline."""
    return add_machine_type(add_machine_fleet_id(add_idle_proxy(_clean_ai4i(df))))


def prepare_ccpp_sequence_arrays(
    df: pd.DataFrame,
    seq_len: int = 24,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pseudo-time windows from sorted ambient conditions for sequence models (LSTM demo).
    Returns X (n, seq_len, n_features), y (n,) with temporal hold-out on the tail.
    """
    df = _clean_ccpp(df)
    df = df.sort_values(by=CCPP_FEATURE_COLS).reset_index(drop=True)
    X = df[CCPP_FEATURE_COLS].values
    y = df[CCPP_TARGET].values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xw, yw = [], []
    for i in range(len(Xs) - seq_len):
        Xw.append(Xs[i : i + seq_len])
        yw.append(y[i + seq_len])
    Xw = np.asarray(Xw, dtype=np.float32)
    yw = np.asarray(yw, dtype=np.float32)
    cut = int(len(Xw) * (1.0 - test_frac))
    return Xw[:cut], Xw[cut:], yw[:cut], yw[cut:]


def prepare_ai4i_splits(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    np.ndarray,
    pd.DataFrame,
    SimpleImputer,
]:
    """
    Returns scaled train/test matrices, labels, test row indices into `df_full`,
    and the fully processed dataframe (for dashboards / insights).
    """
    df_full = preprocess_ai4i_full(df)
    X = df_full[AI4I_SENSOR_COLS].values
    y = df_full["failure_or_mode"].values.astype(int)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    idx = np.arange(len(df_full))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_imp, y, idx, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    feature_names = [c.replace(" ", "_").replace("[", "").replace("]", "") for c in AI4I_SENSOR_COLS]
    return (
        pd.DataFrame(X_tr_s, columns=feature_names),
        pd.DataFrame(X_te_s, columns=feature_names),
        y_tr,
        y_te,
        scaler,
        idx_te,
        df_full,
        imputer,
    )
