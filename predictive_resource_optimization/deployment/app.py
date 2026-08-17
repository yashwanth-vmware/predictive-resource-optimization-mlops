import os
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

# =========================================================================
# Page config + light custom CSS to match the standalone dashboard's look
# =========================================================================
st.set_page_config(
    page_title="Predictive Resource Optimization — Decision Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .kpi-card {
        background: #F5F4F0; border-radius: 10px; padding: 16px;
        text-align: left;
    }
    .kpi-label { font-size: 13px; color: #6B6A64; margin: 0 0 4px 0; }
    .kpi-value { font-size: 24px; font-weight: 600; margin: 0; }
    .section-label { font-size: 14px; color: #6B6A64; margin: 24px 0 8px 0; }
    .risk-critical { color: #B3361D; }
    .risk-success { color: #1B7A4D; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# Config
# =========================================================================
HF_MODEL_REPO_ID = os.getenv(
    "HF_MODEL_REPO_ID", "Yash0204/mlops-infra-predictive-optimization-final-rf"
).strip()

RESOURCES = ["cpu", "memory"]
HORIZONS = ["5min", "15min", "30min"]

AVAILABLE_FEATURES = [
    "cpu_cores", "cpu_capacity_mhz", "cpu_usage_mhz", "cpu_usage_pct",
    "memory_provisioned_kb", "memory_usage_kb", "memory_usage_pct",
    "disk_read_kbps", "disk_write_kbps", "network_received_kbps", "network_transmitted_kbps",
    "disk_total_kbps", "network_total_kbps", "cpu_memory_interaction",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "cpu_usage_pct_lag_1", "cpu_usage_pct_lag_3", "cpu_usage_pct_lag_6",
    "memory_usage_pct_lag_1", "memory_usage_pct_lag_3", "memory_usage_pct_lag_6",
    "cpu_usage_pct_rolling_mean_3", "cpu_usage_pct_rolling_mean_6", "cpu_usage_pct_rolling_mean_12",
    "cpu_usage_pct_rolling_std_3", "cpu_usage_pct_rolling_std_6", "cpu_usage_pct_rolling_std_12",
    "memory_usage_pct_rolling_mean_3", "memory_usage_pct_rolling_mean_6", "memory_usage_pct_rolling_mean_12",
    "memory_usage_pct_rolling_std_3", "memory_usage_pct_rolling_std_6", "memory_usage_pct_rolling_std_12",
    "cpu_usage_pct_diff_1", "memory_usage_pct_diff_1",
]

# Representative VM profiles for the Live Prediction picker -- illustrative
# resource states (not raw dataset rows), built to be internally consistent
# across the model's most important features. See Final Report Limitations.
VM_PROFILES = {
    "VM 118 (idle)":     {"cpu": 1.8,   "cpuLag": 2.1,   "cpuRoll": 2.0,   "mem": 9.4,  "memLag": 9.8,  "cores": 4, "capacity": 8000,  "vol": 0.4},
    "VM 402 (normal)":   {"cpu": 34.5,  "cpuLag": 33.0,  "cpuRoll": 33.8,  "mem": 28.7, "memLag": 27.9, "cores": 4, "capacity": 8000,  "vol": 1.5},
    "VM 174 (busy)":     {"cpu": 81.4,  "cpuLag": 80.2,  "cpuRoll": 80.8,  "mem": 55.3, "memLag": 54.5, "cores": 8, "capacity": 16000, "vol": 2.2},
    "VM 250 (critical)": {"cpu": 102.8, "cpuLag": 101.5, "cpuRoll": 102.2, "mem": 24.3, "memLag": 23.8, "cores": 4, "capacity": 8000,  "vol": 3.0},
    "VM 267 (critical)": {"cpu": 102.2, "cpuLag": 101.0, "cpuRoll": 101.7, "mem": 42.7, "memLag": 41.8, "cores": 8, "capacity": 16000, "vol": 3.0},
}

# Real numbers from the Final Report (Tables 5 and 9) -- used for the KPI
# cards, risk-tier bar, RMSE chart, and detection chart, which are the same
# regardless of which model/horizon is selected in Live Prediction below.
REAL_RMSE_CPU = {
    "5 min":  {"Persistence": 6.32, "Linear Regression": 6.14, "Random Forest": 5.10, "XGBoost": 5.12},
    "15 min": {"Persistence": 11.22, "Linear Regression": 10.81, "Random Forest": 9.55, "XGBoost": 9.59},
    "30 min": {"Persistence": 14.67, "Linear Regression": 13.86, "Random Forest": 12.42, "XGBoost": 12.61},
}
REAL_DETECTION = {
    "CPU 5min": {"precision": 96.7, "recall": 94.1}, "CPU 15min": {"precision": 93.7, "recall": 83.7},
    "CPU 30min": {"precision": 91.4, "recall": 74.5}, "Mem 5min": {"precision": 74.4, "recall": 9.4},
    "Mem 15min": {"precision": 0, "recall": 0}, "Mem 30min": {"precision": 0, "recall": 0},
}
RISK_TIERS = {"Critical": 36, "High": 19, "Normal": 175, "Rightsizing candidate": 267}
WATCHLIST = [
    ("VM 250", 102.8, 24.3), ("VM 267", 102.2, 42.7), ("VM 211", 102.0, 27.9),
    ("VM 226", 101.7, 25.9), ("VM 224", 101.7, 30.5), ("VM 227", 101.5, 25.4),
    ("VM 220", 100.8, 31.8), ("VM 256", 100.4, 21.0),
]


def build_feature_row(vm):
    row = {name: 0.0 for name in AVAILABLE_FEATURES}
    row.update({
        "cpu_usage_pct": vm["cpu"], "cpu_usage_pct_lag_1": vm["cpuLag"],
        "cpu_usage_pct_lag_3": vm["cpuRoll"] + (vm["cpu"] - vm["cpuRoll"]) * 0.6,
        "cpu_usage_pct_lag_6": vm["cpuRoll"] + (vm["cpu"] - vm["cpuRoll"]) * 0.3,
        "cpu_usage_pct_rolling_mean_3": vm["cpuRoll"], "cpu_usage_pct_rolling_mean_6": vm["cpuRoll"],
        "cpu_usage_pct_rolling_mean_12": vm["cpuRoll"], "cpu_usage_pct_rolling_std_3": vm["vol"],
        "cpu_usage_pct_rolling_std_6": vm["vol"] * 1.1, "cpu_usage_pct_rolling_std_12": vm["vol"] * 1.2,
        "cpu_usage_pct_diff_1": vm["cpu"] - vm["cpuLag"], "memory_usage_pct": vm["mem"],
        "memory_usage_pct_lag_1": vm["memLag"], "memory_usage_pct_rolling_mean_3": vm["mem"],
        "memory_usage_pct_rolling_mean_6": vm["mem"], "memory_usage_pct_rolling_mean_12": vm["mem"],
        "memory_usage_pct_rolling_std_3": vm["vol"] * 0.5, "memory_usage_pct_diff_1": vm["mem"] - vm["memLag"],
        "cpu_cores": vm["cores"], "cpu_capacity_mhz": vm["capacity"],
        "cpu_usage_mhz": (vm["cpu"] / 100) * vm["capacity"],
        "cpu_memory_interaction": (vm["cpu"] * vm["mem"]) / 100,
        "memory_usage_kb": (vm["mem"] / 100) * 8_388_608,
    })
    return row


def classify_risk(predicted_pct):
    if predicted_pct >= 90:
        return "Critical", "error"
    if predicted_pct >= 80:
        return "High", "warning"
    if predicted_pct <= 20:
        return "Rightsizing candidate", "success"
    return "Normal", "info"


@st.cache_resource(show_spinner=True)
def load_model(repo_id: str, filename: str):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_metrics(repo_id: str):
    try:
        metrics_path = hf_hub_download(repo_id=repo_id, filename="metrics.json", repo_type="model")
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


# =========================================================================
# Header
# =========================================================================
st.title("Predictive Resource Optimization — Decision Dashboard")
st.caption("AI-Driven Predictive Resource Optimization for On-Premises Virtualized Enterprise Data Centers (QM640 Capstone)")

st.info(
    "**Data note:** KPI cards, risk tiers, RMSE chart, and detection chart use the real metrics "
    "reported in the Final Report (Random Forest vs. baselines on the actual Bitbrains-derived "
    "evaluation). The **Live Prediction** panel is different: it calls your actual trained Random "
    "Forest model, loaded live from the Hugging Face Model Hub, and runs genuine inference."
)

# =========================================================================
# KPI cards
# =========================================================================
k1, k2, k3, k4 = st.columns(4)
for col, label, value, cls in [
    (k1, "VMs scored", "497", ""),
    (k2, "Best model", "Random Forest", ""),
    (k3, "RMSE vs baseline", "-10 to -41%", "risk-success"),
    (k4, "Critical VMs now", "36", "risk-critical"),
]:
    col.markdown(
        f'<div class="kpi-card"><p class="kpi-label">{label}</p>'
        f'<p class="kpi-value {cls}">{value}</p></div>',
        unsafe_allow_html=True,
    )

# =========================================================================
# Risk tier bar
# =========================================================================
st.markdown('<p class="section-label">VM risk tiers — 30-minute forecast</p>', unsafe_allow_html=True)
tier_colors = {"Critical": "#E24B4A", "High": "#EF9F27", "Normal": "#D3D1C7", "Rightsizing candidate": "#5DCAA5"}
tier_fig = go.Figure()
for name, count in RISK_TIERS.items():
    tier_fig.add_trace(go.Bar(
        x=[count], y=["VMs"], orientation="h", name=f"{name} · {count}",
        marker_color=tier_colors[name], hovertemplate=f"{name}: {count}<extra></extra>",
    ))
tier_fig.update_layout(
    barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0),
    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.6),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
)
st.plotly_chart(tier_fig, use_container_width=True)

# =========================================================================
# RMSE chart
# =========================================================================
st.markdown('<p class="section-label">Model RMSE by forecast horizon — CPU</p>', unsafe_allow_html=True)
rmse_fig = go.Figure()
model_colors = {"Persistence": "#898781", "Linear Regression": "#eda100", "Random Forest": "#1baf7a", "XGBoost": "#2a78d6"}
horizons_order = ["5 min", "15 min", "30 min"]
for model_name, color in model_colors.items():
    rmse_fig.add_trace(go.Bar(
        x=horizons_order, y=[REAL_RMSE_CPU[h][model_name] for h in horizons_order],
        name=model_name, marker_color=color,
    ))
rmse_fig.update_layout(barmode="group", height=320, yaxis_title="RMSE", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(rmse_fig, use_container_width=True)

# =========================================================================
# Bottleneck detection chart
# =========================================================================
st.markdown('<p class="section-label">Bottleneck detection: CPU vs memory (Random Forest, 80% threshold)</p>', unsafe_allow_html=True)
det_fig = go.Figure()
det_labels = list(REAL_DETECTION.keys())
det_fig.add_trace(go.Bar(x=det_labels, y=[REAL_DETECTION[k]["precision"] for k in det_labels], name="Precision", marker_color="#2a78d6"))
det_fig.add_trace(go.Bar(x=det_labels, y=[REAL_DETECTION[k]["recall"] for k in det_labels], name="Recall", marker_color="#eb6834"))
det_fig.update_layout(barmode="group", height=320, yaxis_title="%", yaxis_range=[0, 100], margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(det_fig, use_container_width=True)

# =========================================================================
# Live Prediction (real model, real inference)
# =========================================================================
st.markdown('<p class="section-label">Live prediction — REAL MODEL, LOADED FROM HUGGING FACE MODEL HUB</p>', unsafe_allow_html=True)

with st.container(border=True):
    sel1, sel2, sel3 = st.columns(3)
    resource = sel1.selectbox("Resource", RESOURCES)
    horizon = sel2.selectbox("Forecast horizon", HORIZONS)
    vm_name = sel3.selectbox("Select VM", list(VM_PROFILES.keys()))
    vm = VM_PROFILES[vm_name]

    model_filename = f"random_forest_{resource}_{horizon}.joblib"
    try:
        model = load_model(HF_MODEL_REPO_ID, model_filename)
        metrics_data = load_metrics(HF_MODEL_REPO_ID)
        combo_metrics = (metrics_data or {}).get("combinations", {}).get(f"{resource}_{horizon}")
        if combo_metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE (holdout)", f"{combo_metrics['RMSE']:.2f}")
            m2.metric("R²", f"{combo_metrics['R2']:.3f}")
            m3.metric("WAPE", f"{combo_metrics['WAPE_pct']:.1f}%")
    except Exception as e:
        st.error("Failed to load model from Hugging Face Model Hub.")
        st.code(str(e))
        st.stop()

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Current CPU usage", f"{vm['cpu']:.1f}%")
    u2.metric("Current memory usage", f"{vm['mem']:.1f}%")
    u3.metric("CPU cores", vm["cores"])
    u4.metric("CPU capacity", f"{vm['capacity']:,} MHz")

    if st.button("Predict", type="primary"):
        row = build_feature_row(vm)
        X = pd.DataFrame([row])[AVAILABLE_FEATURES]
        prediction = model.predict(X)[0]
        risk_label, risk_kind = classify_risk(prediction)
        target_name = "CPU" if resource == "cpu" else "memory"
        message = f"**{vm_name} — Predicted {target_name} utilization ({horizon} ahead): {prediction:.2f}%**  \nRisk: **{risk_label}**"
        getattr(st, risk_kind)(message)
        st.caption(
            "Note: predicted value can legitimately differ from current usage -- that is what a "
            "forecast is for. Random Forest moderates toward patterns seen more often in training, "
            "so it can under-react to a VM at a rare, sustained extreme reading. See the Final "
            "Report's Limitations section for details."
        )

# =========================================================================
# Watchlist table
# =========================================================================
st.markdown('<p class="section-label">Top watchlist — highest predicted CPU, next 30 minutes</p>', unsafe_allow_html=True)
watchlist_df = pd.DataFrame(WATCHLIST, columns=["VM ID", "Predicted CPU", "Predicted memory"])
watchlist_df["Predicted CPU"] = watchlist_df["Predicted CPU"].map(lambda v: f"{v:.1f}%")
watchlist_df["Predicted memory"] = watchlist_df["Predicted memory"].map(lambda v: f"{v:.1f}%")
watchlist_df["Risk"] = "Critical"
st.dataframe(watchlist_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Source: QM640 Final Report (Results section, Tables 5 and 9) and the executed capstone "
    "notebook's decision-support output. Live Prediction calls your real model on Hugging Face."
)
