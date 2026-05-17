"""
WattWise AI: Industrial Energy Dashboard
Premium dark theme with exact design match
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from industrial_energy_optimizer.config import (
    DEFAULT_TARIFF_OFFPEAK_INR_PER_KWH,
    DEFAULT_TARIFF_PEAK_INR_PER_KWH,
)
from industrial_energy_optimizer.data_io import load_ai4i, load_ccpp
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

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="WattWise AI: Industrial Energy Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ========== CUSTOM CSS - EXACT DESIGN MATCH ==========
st.markdown(
    """
    <style>
    /* Root styling */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f35 100%);
        color: #ffffff;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f35 100%);
        padding: 20px;
    }
    
    /* Main title */
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    
    /* Subheader */
    h2 {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* Cards */
    [data-testid="stVerticalBlock"] > div > div {
        background: linear-gradient(135deg, rgba(20, 30, 60, 0.6) 0%, rgba(35, 45, 75, 0.6) 100%);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(20, 30, 60, 0.8) 0%, rgba(35, 45, 75, 0.8) 100%);
        border: 1px solid rgba(255, 165, 0, 0.4);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #ffa500;
        font-size: 32px;
        font-weight: 700;
    }
    
    /* Metric label */
    [data-testid="stMetricLabel"] {
        color: #ffffff;
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ffa500 0%, #ffb84d 100%);
        color: #000000;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 10px 24px;
        box-shadow: 0 4px 12px rgba(255, 165, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffb84d 0%, #ffc966 100%);
        box-shadow: 0 6px 16px rgba(255, 165, 0, 0.6);
        transform: translateY(-2px);
    }
    
    /* Column dividers */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, rgba(255, 165, 0, 0.3) 0%, transparent 100%);
        margin: 20px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f35 100%);
        border-right: 1px solid rgba(255, 165, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    # ========== TITLE ==========
    st.markdown(
        "<h1>WattWise AI: Industrial Energy Dashboard</h1>",
        unsafe_allow_html=True,
    )
    
    # ========== LOAD DATA & TRAIN ==========
    with st.spinner("Loading UCI datasets…"):
        df_ccpp, df_ai = _load_data()

    with st.spinner("Training models…"):
        bundle, lstm, scaler_ai, imputer_ai, idx_te, df_full, X_itr, X_ite = _train_bundle(
            df_ccpp, df_ai, 42
        )

    # ========== PREPARE DATA ==========
    df_scored = attach_fleet_anomaly_scores(
        df_full,
        imputer_ai,
        scaler_ai,
        bundle.machine_rf,
        bundle.feature_names_ai,
    )
    machine_tbl = build_machine_wise_table(df_scored)
    
    y_hat = (
        bundle.y_ccpp_pred_xgb
        if bundle.y_ccpp_pred_xgb is not None
        else bundle.y_ccpp_pred_rf
    )
    cost_df, total_inr = macro_plant_cost_curve(y_hat, 12.0, 7.5)
    hotspots = build_machine_hotspots(df_scored, 12.0, 7.5)
    actions = recommended_actions(hotspots, total_inr)

    # ========== METRICS HEADER ==========
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Calculate metrics
    idle_loss = hotspots["wastage_inr_total_est"].sum()
    efficiency = bundle.plant_metrics.get('r2_rf', 0.5) * 100
    daily_savings = sum([a.est_savings_inr for a in actions])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(200, 50, 50, 0.3) 0%, rgba(220, 70, 70, 0.3) 100%); 
                        border: 2px solid #ff4444; border-radius: 8px; padding: 16px; text-align: center;'>
                <div style='color: #ff6b6b; font-size: 12px; font-weight: 600; margin-bottom: 8px;'>
                    ❌ Idle Losses Today:
                </div>
                <div style='color: #ffa500; font-size: 28px; font-weight: 700;'>
                    ₹ {idle_loss:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(50, 150, 50, 0.3) 0%, rgba(70, 170, 70, 0.3) 100%); 
                        border: 2px solid #44ff44; border-radius: 8px; padding: 16px; text-align: center;'>
                <div style='color: #66ff66; font-size: 12px; font-weight: 600; margin-bottom: 8px;'>
                    ✓ Efficiency:
                </div>
                <div style='color: #66ff66; font-size: 28px; font-weight: 700;'>
                    {efficiency:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, rgba(50, 150, 50, 0.3) 0%, rgba(70, 170, 70, 0.3) 100%); 
                        border: 2px solid #44ff44; border-radius: 8px; padding: 16px; text-align: center;'>
                <div style='color: #66ff66; font-size: 12px; font-weight: 600; margin-bottom: 8px;'>
                    ⬇️ Potential Savings:
                </div>
                <div style='color: #ffa500; font-size: 28px; font-weight: 700;'>
                    ₹ {daily_savings:,.0f}/day
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # ========== MAIN GRID ==========
    col_left, col_right = st.columns(2)

    # ========== LEFT COLUMN ==========
    with col_left:
        # Energy Demand Forecast
        st.markdown(
            "<h2 style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 24px;'>▶️</span> Energy Demand Forecast</h2>",
            unsafe_allow_html=True,
        )
        
        fc = build_forecast_curve(y_hat)
        fig_demand = go.Figure()
        
        # Area chart for actual demand
        fig_demand.add_trace(go.Scatter(
            x=fc['scenario_index'],
            y=fc['forecast_mw'],
            fill='tozeroy',
            fillcolor='rgba(135, 206, 250, 0.3)',
            line=dict(color='#4169E1', width=2),
            name='Actual Demand'
        ))
        
        # Dashed line for power (simulating forecast)
        peak_threshold = np.percentile(y_hat, 75)
        fig_demand.add_trace(go.Scatter(
            x=fc['scenario_index'],
            y=[peak_threshold] * len(fc),
            line=dict(color='#ffa500', width=3, dash='dash'),
            name='Power Ahuumt'
        ))
        
        # Peak alert annotation
        peak_idx = np.argmax(y_hat)
        fig_demand.add_annotation(
            x=peak_idx,
            y=y_hat[peak_idx],
            text="Peak Alert!",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#ff6b35",
            ax=-40,
            ay=-40,
            bgcolor="#ff6b35",
            bordercolor="#ff6b35",
            borderwidth=2,
            font=dict(color="white", size=12),
        )
        
        fig_demand.update_layout(
            height=350,
            template="plotly_dark",
            plot_bgcolor="rgba(15, 30, 60, 0.3)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,165,0,0.1)"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,165,0,0.1)"),
            font=dict(color="#ffffff", size=11),
        )
        
        st.plotly_chart(fig_demand, use_container_width=True)
        
        # Trend & Cost info
        col_trend1, col_trend2 = st.columns(2)
        with col_trend1:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <span style='color: #66ff66; font-weight: 600; font-size: 14px;'>✓ Trend Efficiency:</span>
                    <div style='color: #ffa500; font-size: 20px; font-weight: 700;'>{efficiency:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_trend2:
            peak_cost = float(np.sum(cost_df[cost_df["is_peak_window"] == 1]["cost_inr"]))
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <span style='color: #ff6b6b; font-weight: 600; font-size: 14px;'>✕ Peak Cost:</span>
                    <div style='color: #ffa500; font-size: 20px; font-weight: 700;'>₹ {peak_cost:,.0f} / day</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Cost Breakdown
        st.markdown(
            "<h2 style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 24px;'>🎯</span> Cost Breakdown</h2>",
            unsafe_allow_html=True,
        )
        
        operational_cost = float(cost_df[cost_df["is_peak_window"] == 0]["cost_inr"].sum())
        maintenance = idle_loss * 0.10
        total_cost = idle_loss + operational_cost + maintenance
        
        # Donut chart
        fig_cost = go.Figure(data=[go.Pie(
            labels=['Idle Losses', 'Operational', 'Maintenance'],
            values=[idle_loss, operational_cost, maintenance],
            hole=0.4,
            marker=dict(colors=['#ff6b6b', '#4169E1', '#ffa500']),
            textposition='inside',
            textinfo='label',
            hovertemplate='<b>%{label}</b><br>₹ %{value:,.0f}<extra></extra>',
        )])
        
        fig_cost.update_layout(
            height=320,
            template="plotly_dark",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=True,
            legend=dict(x=1.05, y=1, bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#ffffff", size=11),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Cost breakdown details
        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            st.markdown(
                f"""
                <div style='background: rgba(20, 30, 60, 0.8); border-left: 3px solid #ff6b6b; padding: 12px; border-radius: 6px;'>
                    <div style='color: #ff6b6b; font-size: 11px; font-weight: 600;'>Idle Losses</div>
                    <div style='color: #ffa500; font-size: 18px; font-weight: 700;'>₹ {idle_loss:,.0f}</div>
                </div>
                <div style='background: rgba(20, 30, 60, 0.8); border-left: 3px solid #4169E1; padding: 12px; border-radius: 6px; margin-top: 8px;'>
                    <div style='color: #4169E1; font-size: 11px; font-weight: 600;'>Operational</div>
                    <div style='color: #ffa500; font-size: 18px; font-weight: 700;'>₹ {operational_cost:,.0f}</div>
                </div>
                <div style='background: rgba(20, 30, 60, 0.8); border-left: 3px solid #ffa500; padding: 12px; border-radius: 6px; margin-top: 8px;'>
                    <div style='color: #ffa500; font-size: 11px; font-weight: 600;'>Maintenance</div>
                    <div style='color: #ffa500; font-size: 18px; font-weight: 700;'>₹ {maintenance:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_cost2:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, rgba(255, 165, 0, 0.2) 0%, rgba(255, 180, 77, 0.2) 100%); 
                            border: 2px solid #ffa500; border-radius: 8px; padding: 16px; height: 100%;'>
                    <div style='color: #ffa500; font-size: 11px; font-weight: 600; margin-bottom: 8px;'>Total Energy Cost</div>
                    <div style='color: #ffffff; font-size: 28px; font-weight: 700;'>₹ {total_cost:,.0f}</div>
                    <div style='color: #cccccc; font-size: 10px; margin-top: 8px;'>/ day</div>
                    <hr style='border: 0; height: 1px; background: rgba(255, 165, 0, 0.3); margin: 12px 0;'>
                    <div style='color: #66ff66; font-size: 11px; font-weight: 600; margin-bottom: 4px;'>Potential Savings</div>
                    <div style='color: #66ff66; font-size: 20px; font-weight: 700;'>₹ {daily_savings:,.0f}</div>
                    <div style='color: #cccccc; font-size: 10px;'>/ day</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========== RIGHT COLUMN ==========
    with col_right:
        # Machine Anomalies & Costs
        st.markdown(
            "<h2 style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 24px;'>⚙️</span> Machine Anomalies & Costs</h2>",
            unsafe_allow_html=True,
        )
        
        # Top machines table
        top_machines = hotspots.head(3).copy()
        
        for idx, (_, machine) in enumerate(top_machines.iterrows()):
            if idx == 0:
                bg_color = "rgba(200, 50, 50, 0.3)"
                border_color = "#ff4444"
            elif idx == 1:
                bg_color = "rgba(220, 100, 30, 0.3)"
                border_color = "#ff8844"
            else:
                bg_color = "rgba(200, 150, 0, 0.3)"
                border_color = "#ffaa44"
            
            machine_name = f"Machine {int(machine['machine_id'])}"
            status = "Idle" if machine['idle_hours'] > 10 else "Overheating" if machine['mean_air_K'] > 310 else "Warning"
            wastage = machine['wastage_inr_total_est']
            
            st.markdown(
                f"""
                <div style='background: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; 
                            padding: 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='color: #ffffff; font-weight: 600;'>{machine_name}</div>
                        <div style='color: #cccccc; font-size: 12px;'>{status}</div>
                    </div>
                    <div style='color: #ffa500; font-weight: 700; font-size: 18px;'>₹ {wastage:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Failure causes
        st.markdown(
            "<div style='font-weight: 600; color: #ffffff; margin-bottom: 12px;'>Uuday Grauses:</div>",
            unsafe_allow_html=True,
        )
        
        failures = [
            ("🔥 High Temp", "#ff6b6b"),
            ("⚡ Excess Vibration", "#ffa500"),
            ("⏱️ Idle Running", "#ffaa44"),
        ]
        
        for failure, color in failures:
            st.markdown(
                f"""
                <div style='background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2); 
                            border-left: 3px solid {color}; padding: 10px 12px; border-radius: 6px; margin-bottom: 8px;'>
                    <span style='color: {color}; font-weight: 600;'>{failure}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Optimization Recommendations
        st.markdown(
            "<h2 style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 24px;'>⚡</span> Optimization Recommendations</h2>",
            unsafe_allow_html=True,
        )
        
        recommendations_list = [
            ("Shift Compressor A to Night Shift", "Save ₹ 20K / Day", "#ffa500"),
            ("Schedule Maintenance for Motor B", "Prevent Overheating", "#66ff66"),
            ("Optimize Pump C Performance", "Reduce Vibration", "#4169E1"),
        ]
        
        for rec_title, rec_action, rec_color in recommendations_list:
            st.markdown(
                f"""
                <div style='margin-bottom: 12px;'>
                    <div style='color: {rec_color}; font-size: 13px; font-weight: 600; margin-bottom: 6px;'>● {rec_title}</div>
                    <button style='background: {rec_color}; color: #000000; border: none; border-radius: 6px; 
                                   padding: 8px 16px; font-weight: 700; cursor: pointer; width: 100%; font-size: 12px;'>
                        {rec_action}
                    </button>
                </div>
                """,
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
