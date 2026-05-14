"""Macro (plant) + micro (fleet) integration and ₹ cost mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from industrial_energy_optimizer.config import MW_TO_KWH_PER_HOUR
from industrial_energy_optimizer.preprocess import (
    AI4I_SENSOR_COLS,
    build_machine_wise_table,
)


def kwh_from_mw_hour(mw: np.ndarray | float) -> np.ndarray:
    """Hourly plant output in MW → electrical energy in kWh (1 h window)."""
    return np.asarray(mw, dtype=float) * MW_TO_KWH_PER_HOUR


def inr_from_kwh(kwh: np.ndarray | float, tariff_inr_per_kwh: float) -> np.ndarray:
    return np.asarray(kwh, dtype=float) * float(tariff_inr_per_kwh)


def peak_mask_from_forecast(
    forecast_mw: np.ndarray, quantile: float = 0.75
) -> np.ndarray:
    """Mark hours in the top `quantile` of predicted output as peak stress."""
    thr = float(np.quantile(forecast_mw, quantile))
    return forecast_mw >= thr


def attach_fleet_anomaly_scores(
    df_full: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    clf: RandomForestClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    X = imputer.transform(df_full[AI4I_SENSOR_COLS].values)
    Xs = scaler.transform(X)
    Xdf = pd.DataFrame(Xs, columns=feature_names)
    out = df_full.copy()
    out["anomaly_proba"] = clf.predict_proba(Xdf)[:, 1].astype(float)
    return out


@dataclass
class ActionItem:
    priority: int
    title: str
    detail: str
    est_savings_inr: float


def build_machine_hotspots(
    df_scored: pd.DataFrame,
    tariff_peak: float,
    tariff_offpeak: float,
    peak_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Machine-level wastage proxy: idle-like hours weighted by tariff,
    scaled by anomaly probability (micro risk).
    """
    agg = build_machine_wise_table(df_scored)
    df_work = df_scored.copy()
    thr = float(np.quantile(df_work["anomaly_proba"], 1.0 - peak_fraction))
    df_work["_peak_proxy"] = (df_work["anomaly_proba"] >= thr).astype(int)
    df_work["_idle_peak"] = (
        (df_work["idle_like"] == 1) & (df_work["_peak_proxy"] == 1)
    ).astype(int)
    idle_peak = (
        df_work.groupby(["machine_id", "Type"], as_index=False)["_idle_peak"]
        .sum()
        .rename(columns={"_idle_peak": "idle_at_peak_proxy"})
    )
    agg = agg.merge(idle_peak, on=["machine_id", "Type"], how="left")
    agg["idle_at_peak_proxy"] = agg["idle_at_peak_proxy"].fillna(0).astype(int)
    mpro = (
        df_work.groupby(["machine_id", "Type"], as_index=False)["anomaly_proba"]
        .mean()
        .rename(columns={"anomaly_proba": "mean_anomaly_proba"})
    )
    agg = agg.merge(mpro, on=["machine_id", "Type"], how="left")
    agg["mean_anomaly_proba"] = agg["mean_anomaly_proba"].fillna(0.0)
    agg["wastage_kwh_est"] = agg["total_energy_kwh_est"] * agg["mean_anomaly_proba"]
    agg["wastage_inr_peak_est"] = inr_from_kwh(
        agg["idle_at_peak_proxy"] * agg["mean_power_kw"] * 1.0, tariff_peak
    )
    agg["wastage_inr_offpeak_est"] = inr_from_kwh(
        (agg["idle_hours"] - agg["idle_at_peak_proxy"]).clip(lower=0)
        * agg["mean_power_kw"],
        tariff_offpeak,
    )
    agg["wastage_inr_total_est"] = (
        agg["wastage_inr_peak_est"] + agg["wastage_inr_offpeak_est"]
    )
    return agg.sort_values("wastage_inr_total_est", ascending=False)


def narrative_insight(
    machine_id: int,
    machine_type: str,
    wastage_inr: float,
    idle_peak: int,
) -> str:
    ex = min(wastage_inr, 30_000.0)
    return (
        f"Machine {machine_id} ({machine_type}) — idle / abnormal pattern estimated at "
        f"₹{wastage_inr:,.0f} annualized proxy ({idle_peak} idle-like rows co-occurring with "
        f"fleet stress). Example scale: ₹{ex:,.0f} during peak-stress windows."
    )


def recommended_actions(
    hotspots: pd.DataFrame,
    plant_peak_cost_inr: float,
    max_items: int = 8,
) -> List[ActionItem]:
    items: List[ActionItem] = []
    if plant_peak_cost_inr > 0:
        items.append(
            ActionItem(
                priority=1,
                title="Stagger peak plant load",
                detail="Macro forecast shows high-output hours; shift non-critical shop loads off those windows.",
                est_savings_inr=plant_peak_cost_inr * 0.05,
            )
        )
    for _, row in hotspots.head(max_items).iterrows():
        mid = int(row["machine_id"])
        mt = str(row["Type"])
        w = float(row["wastage_inr_total_est"])
        ip = int(row["idle_at_peak_proxy"])
        items.append(
            ActionItem(
                priority=2 if w > 50_000 else 3,
                title=f"Investigate fleet unit {mid} (type {mt})",
                detail=narrative_insight(mid, mt, w, ip),
                est_savings_inr=min(w * 0.15, 500_000.0),
            )
        )
    items.sort(key=lambda a: (a.priority, -a.est_savings_inr))
    return items[:max_items]


def macro_plant_cost_curve(
    forecast_mw: np.ndarray,
    tariff_peak: float,
    tariff_offpeak: float,
    peak_q: float = 0.75,
) -> Tuple[pd.DataFrame, float]:
    """Hourly cost if we map high-output hours to peak tariff."""
    m = peak_mask_from_forecast(forecast_mw, quantile=peak_q)
    kwh = kwh_from_mw_hour(forecast_mw)
    rate = np.where(m, tariff_peak, tariff_offpeak)
    cost = kwh * rate
    df = pd.DataFrame(
        {
            "hour_rank": np.arange(len(forecast_mw)),
            "forecast_mw": forecast_mw,
            "is_peak_window": m.astype(int),
            "kwh": kwh,
            "cost_inr": cost,
        }
    )
    return df, float(np.sum(cost))
