"""Macro (plant) + micro (machines) integration for actionable insights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_energy_optimizer.economics import machine_idle_wastage_inr


def assign_peak_by_macro_quantile(
    macro_pred_mw: np.ndarray, upper_q: float = 0.8
) -> tuple[np.ndarray, float]:
    """Peak hours bucket: top (1-upper_q) quantile of macro predicted load."""
    thr = float(np.quantile(macro_pred_mw, upper_q))
    is_peak = (macro_pred_mw >= thr).astype(int)
    return is_peak, thr


def align_machines_to_macro_scenario(
    df_ai: pd.DataFrame,
    macro_pred_mw: np.ndarray,
) -> pd.DataFrame:
    """
    Without shared timestamps, map each machine row to a macro scenario bucket
    deterministically from UDI for a stable demo integration.
    """
    out = df_ai.copy()
    n_macro = len(macro_pred_mw)
    idx = (out["UDI"].astype(int).values % n_macro).astype(int)
    out["macro_bucket"] = idx
    out["macro_pred_mw"] = macro_pred_mw[idx]
    return out


def build_actionable_insights(
    df_ai: pd.DataFrame,
    macro_pred_mw: np.ndarray,
    anomaly_score: np.ndarray,
    anomaly_flag: np.ndarray,
    peak_tariff: float,
    offpeak_tariff: float,
    top_n: int = 15,
) -> pd.DataFrame:
    """Rows flagged abnormal or idle-like during macro peak with ₹ impact."""
    df = align_machines_to_macro_scenario(df_ai, macro_pred_mw)
    is_peak, thr = assign_peak_by_macro_quantile(macro_pred_mw, upper_q=0.8)
    df["macro_peak"] = df["macro_bucket"].map(lambda i: int(is_peak[int(i)]))
    df["anomaly_flag"] = anomaly_flag
    df["anomaly_score"] = anomaly_score

    waste_inr = machine_idle_wastage_inr(
        df["power_kw_proxy"].values.astype(float),
        df["macro_peak"].values.astype(int),
        df["idle_like"].values.astype(int),
        hours_each=1.0,
        peak_tariff=peak_tariff,
        offpeak_tariff=offpeak_tariff,
        waste_fraction_when_idle=0.4,
    )
    df["idle_wastage_inr"] = waste_inr

    df["action_priority"] = (
        df["anomaly_flag"].astype(int) * 3
        + df["idle_like"].astype(int) * 2
        + df["macro_peak"].astype(int)
    )
    interesting = df[
        (df["anomaly_flag"] == 1) | ((df["idle_like"] == 1) & (df["macro_peak"] == 1))
    ].copy()
    interesting = interesting.sort_values(
        ["idle_wastage_inr", "anomaly_score"], ascending=False
    )
    cols = [
        "UDI",
        "Product ID",
        "Type",
        "macro_pred_mw",
        "macro_peak",
        "idle_like",
        "anomaly_flag",
        "anomaly_score",
        "idle_wastage_inr",
        "failure_or_mode",
    ]
    interesting = interesting[cols].head(top_n)
    interesting["insight"] = interesting.apply(
        lambda r: _insight_sentence(r, peak_thr_mw=thr, peak_tariff=peak_tariff),
        axis=1,
    )
    return interesting


def _insight_sentence(r: pd.Series, peak_thr_mw: float, peak_tariff: float) -> str:
    parts = []
    if int(r["macro_peak"]) == 1:
        parts.append(
            f"macro load scenario ≥ {peak_thr_mw:.1f} MW (peak-style pricing ₹{peak_tariff:.1f}/kWh)"
        )
    if int(r["idle_like"]) == 1:
        parts.append("idle-like operating pattern (high RPM, low torque vs cohort)")
    if int(r["anomaly_flag"]) == 1:
        parts.append(f"model anomaly risk score {float(r['anomaly_score']):.2f}")
    tail = "; ".join(parts) if parts else "routine"
    wastage = float(r["idle_wastage_inr"])
    return (
        f"Machine UDI {int(r['UDI'])} ({r['Product ID']}, type {r['Type']}): "
        f"estimated idle/stranded energy cost ₹{wastage:,.0f} in mapped peak window; {tail}."
    )
