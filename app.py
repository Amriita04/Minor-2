"""
Industrial Energy Optimizer — Streamlit dashboard (macro plant + micro fleet).
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from industrial_energy_optimizer.config import (
    DEFAULT_TARIFF_OFFPEAK_INR_PER_KWH,
    DEFAULT_TARIFF_PEAK_INR_PER_KWH,
)
from industrial_energy_optimizer.data_io import load_ai4i, load_ccpp
from industrial_energy_optimizer.explainability import explain_anomalies
from industrial_energy_optimizer.integration import (
    attach_fleet_anomaly_scores,
    build_machine_hotspots,
    macro_plant_cost_curve,
    recommended_actions,
)
from industrial_energy_optimizer.modeling import (
    HAS_TF,
    HAS_XGB,
    build_forecast_curve,
    train_models,
    train_plant_lstm,
)
from industrial_energy_optimizer.preprocess import (
    build_machine_wise_table,
    prepare_ai4i_splits,
    prepare_ccpp_splits,
)

st.set_page_config(
    page_title="Industrial Energy Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=True)
def _load_data():
    return load_ccpp(force_download=False), load_ai4i(force_download=False)


@st.cache_resource(show_spinner=True)
def _train_bundle(_ccpp: pd.DataFrame, _ai: pd.DataFrame, rs: int):
    X_ctr, X_cte, y_ctr, y_cte, _ = prepare_ccpp_splits(_ccpp, random_state=rs)
    X_itr, X_ite, y_itr, y_ite, scaler_ai, idx_te, df_full, imputer_ai = (
        prepare_ai4i_splits(_ai, random_state=rs)
    )
    bundle = train_models(
        X_ctr, X_cte, y_ctr, y_cte, X_itr, X_ite, y_itr, y_ite, random_state=rs
    )
    lstm = train_plant_lstm(_ccpp, random_state=rs)
    return bundle, lstm, scaler_ai, imputer_ai, idx_te, df_full, X_itr, X_ite


def main() -> None:
    st.title("Industrial Energy Optimizer (software-only prototype)")
    st.caption(
        "UCI Combined Cycle Power Plant (macro regression) + AI4I 2020 (micro classification). "
        "Costs use **units × tariff** on proxy kWh from MW and mechanical power."
    )

    with st.sidebar:
        st.header("Tariffs (₹/kWh)")
        t_peak = st.number_input(
            "Peak",
            min_value=0.0,
            value=float(DEFAULT_TARIFF_PEAK_INR_PER_KWH),
            step=0.5,
        )
        t_off = st.number_input(
            "Off-peak",
            min_value=0.0,
            value=float(DEFAULT_TARIFF_OFFPEAK_INR_PER_KWH),
            step=0.5,
        )
        rs = st.number_input("Random seed", value=42, step=1)
        run = st.button("Reload data & retrain", type="primary")

    if run:
        st.cache_data.clear()
        st.cache_resource.clear()

    with st.spinner("Loading UCI datasets…"):
        df_ccpp, df_ai = _load_data()

    st.header("1. Data")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Combined Cycle Power Plant")
        st.dataframe(df_ccpp.head(8), use_container_width=True)
        st.write(f"Rows: **{len(df_ccpp):,}** — Features: AT, V, AP, RH → target **PE** (MW).")
    with c2:
        st.subheader("AI4I 2020 Predictive Maintenance")
        st.dataframe(df_ai.head(8), use_container_width=True)
        st.write(
            f"Rows: **{len(df_ai):,}** — Sensors + failure flags; **machine_id** bins UDI into a synthetic fleet."
        )

    bundle, lstm, scaler_ai, imputer_ai, idx_te, df_full, X_itr, X_ite = _train_bundle(
        df_ccpp, df_ai, int(rs)
    )

    st.header("2. Analysis")
    st.subheader("Model metrics (example-driven)")
    mcols = st.columns(4)
    mcols[0].metric("Plant R² (RF)", f"{bundle.plant_metrics.get('r2_rf', 0):.3f}")
    mcols[1].metric(
        "Plant MAE MW (RF)", f"{bundle.plant_metrics.get('mae_rf_mw', 0):.2f}"
    )
    if HAS_XGB and "r2_xgb" in bundle.plant_metrics:
        mcols[2].metric("Plant R² (XGB)", f"{bundle.plant_metrics['r2_xgb']:.3f}")
    else:
        mcols[2].metric("XGBoost", "n/a")
    if lstm.y_pred is not None:
        mcols[3].metric(
            "Plant R² (LSTM)",
            f"{lstm.metrics.get('r2_lstm', 0):.3f}",
        )
    else:
        mcols[3].metric("LSTM", "skipped" if not HAS_TF else "n/a")

    st.write(
        "**Classification (machines):** Random Forest accuracy "
        f"{bundle.machine_metrics['accuracy_rf']:.3f}; "
        f"Logistic Regression accuracy {bundle.machine_metrics['accuracy_lr']:.3f}."
    )
    with st.expander("Classification report (RF)"):
        st.text(bundle.machine_metrics["report_rf"])

    df_scored = attach_fleet_anomaly_scores(
        df_full,
        imputer_ai,
        scaler_ai,
        bundle.machine_rf,
        bundle.feature_names_ai,
    )
    machine_tbl = build_machine_wise_table(df_scored)
    st.subheader("Machine-wise consumption (aggregated)")
    st.dataframe(machine_tbl.head(12), use_container_width=True)

    st.header("3. Cost mapping (₹)")
    y_hat = (
        bundle.y_ccpp_pred_xgb
        if bundle.y_ccpp_pred_xgb is not None
        else bundle.y_ccpp_pred_rf
    )
    cost_df, total_inr = macro_plant_cost_curve(y_hat, t_peak, t_off)
    st.write(
        f"Example: hourly energy from forecast MW × **1000 kWh/MW** × tariff band → "
        f"scenario **total ₹ {total_inr:,.0f}** over the held-out plant window (illustrative)."
    )
    fig_cost = px.line(
        cost_df,
        x="hour_rank",
        y="cost_inr",
        color="is_peak_window",
        title="Plant-window cost curve (₹) from forecast MW",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    hotspots = build_machine_hotspots(df_scored, t_peak, t_off)
    fig_bar = px.bar(
        hotspots.head(15),
        x="machine_id",
        y="wastage_inr_total_est",
        color="Type",
        title="Top machine wastage hotspots (₹ proxy)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.header("4. Suggestions (macro + micro)")
    actions = recommended_actions(hotspots, total_inr)
    for a in actions:
        st.markdown(
            f"**P{a.priority}** — {a.title} — est. savings **₹{a.est_savings_inr:,.0f}**  \n"
            f"{a.detail}"
        )

    st.subheader("Actionable insight example")
    if len(hotspots):
        r0 = hotspots.iloc[0]
        st.info(
            f"Machine **{int(r0['machine_id'])}** (type **{r0['Type']}**) — idle wastage proxy "
            f"**₹{float(r0['wastage_inr_peak_est']):,.0f}** during fleet stress windows "
            f"(coarse demo; tune tariffs and sensors for production)."
        )

    st.header("5. Explainability (SHAP + LIME)")
    samples, note = explain_anomalies(
        bundle.machine_rf,
        X_itr,
        X_ite,
        idx_te,
        max_samples=4,
    )
    st.caption(note)
    for s in samples:
        st.markdown(
            f"**Row UDI ~ {s.row_index}** — abnormal probability **{s.predicted_proba:.2f}**"
        )
        if s.shap_top_features:
            st.write("SHAP (top magnitudes):", dict(s.shap_top_features))
        if s.lime_weights:
            st.write("LIME weights:", dict(s.lime_weights))

    st.header("6. Dashboard simulation")
    fc = build_forecast_curve(y_hat)
    fig_fc = px.line(
        fc,
        x="scenario_index",
        y="forecast_mw",
        title="Sorted demand / output forecast curve (macro)",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    if lstm.y_pred is not None:
        fig_l = go.Figure()
        fig_l.add_trace(
            go.Scatter(
                y=lstm.y_test,
                mode="markers",
                name="Actual MW",
                marker=dict(size=4, opacity=0.35),
            )
        )
        fig_l.add_trace(
            go.Scatter(
                y=lstm.y_pred,
                mode="markers",
                name="LSTM pred",
                marker=dict(size=4, opacity=0.35),
            )
        )
        fig_l.update_layout(title="LSTM hold-out: actual vs predicted PE (MW)")
        st.plotly_chart(fig_l, use_container_width=True)

    st.success(
        "This system pinpoints industrial electricity wastage in ₹ terms and provides "
        "actionable cost-saving suggestions."
    )


if __name__ == "__main__":
    main()
