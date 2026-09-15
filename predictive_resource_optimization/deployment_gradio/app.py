import io
import json
import os

import gradio as gr
import joblib
import matplotlib
import spaces
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# =========================================================================
# Config -- HF_MODEL_REPO_ID below gets patched by the notebook (Section
# 5.9) to your actual configured repo id, same pattern used for the
# Streamlit app.py in Section 5.3. Reads the Space's own HF_MODEL_REPO_ID
# variable at runtime if set, falling back to the patched default.
# =========================================================================
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "Yash0204/mlops-infra-predictive-optimization-final-rf")

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
        return "Critical", "🔴"
    if predicted_pct >= 80:
        return "High", "🟠"
    if predicted_pct <= 20:
        return "Rightsizing candidate", "🟢"
    return "Normal", "🔵"


# =========================================================================
# Model / metrics loading -- lazy, on-demand, cached in a module-level
# dict so repeated predictions for the same resource/horizon don't
# re-download. Normal huggingface_hub usage -- no browser/Pyodide
# constraints here since this runs as a regular server-side Python app.
# =========================================================================
_model_cache = {}
_metrics_cache = {"data": None, "loaded": False}


def get_model(resource, horizon):
    key = f"{resource}_{horizon}"
    if key not in _model_cache:
        filename = f"random_forest_{resource}_{horizon}.joblib"
        local_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=filename)
        _model_cache[key] = joblib.load(local_path)
    return _model_cache[key]


def get_metrics():
    if not _metrics_cache["loaded"]:
        try:
            local_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename="metrics.json")
            with open(local_path) as f:
                _metrics_cache["data"] = json.load(f)
        except Exception:
            _metrics_cache["data"] = None
        _metrics_cache["loaded"] = True
    return _metrics_cache["data"]


# =========================================================================
# Static charts (Matplotlib)
# =========================================================================
def make_risk_tier_fig():
    fig, ax = plt.subplots(figsize=(7, 1.3))
    colors = {"Critical": "#E24B4A", "High": "#EF9F27", "Normal": "#D3D1C7", "Rightsizing candidate": "#5DCAA5"}
    left = 0
    for name, count in RISK_TIERS.items():
        ax.barh(["VMs"], [count], left=left, color=colors[name], label=f"{name} · {count}")
        left += count
    ax.set_xlim(0, sum(RISK_TIERS.values()))
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def make_rmse_fig():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    horizons_order = ["5 min", "15 min", "30 min"]
    model_colors = {"Persistence": "#898781", "Linear Regression": "#eda100", "Random Forest": "#1baf7a", "XGBoost": "#2a78d6"}
    x = np.arange(len(horizons_order))
    width = 0.2
    for i, (model_name, color) in enumerate(model_colors.items()):
        values = [REAL_RMSE_CPU[h][model_name] for h in horizons_order]
        ax.bar(x + (i - 1.5) * width, values, width, label=model_name, color=color)
    ax.set_xticks(x, horizons_order)
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_detection_fig():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = list(REAL_DETECTION.keys())
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [REAL_DETECTION[k]["precision"] for k in labels], width, label="Precision", color="#2a78d6")
    ax.bar(x + width / 2, [REAL_DETECTION[k]["recall"] for k in labels], width, label="Recall", color="#eb6834")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================================
# Live prediction handler
# =========================================================================
@spaces.GPU(duration=5)
def predict(resource, horizon, vm_name):
    # This app only needs CPU (plain scikit-learn inference) -- the
    # @spaces.GPU decorator above is required by ZeroGPU Spaces to pass
    # their startup check, not because this function actually uses a GPU.
    vm = VM_PROFILES[vm_name]
    usage_md = (
        f"Current CPU usage: **{vm['cpu']:.1f}%** &nbsp;|&nbsp; "
        f"Current memory usage: **{vm['mem']:.1f}%** &nbsp;|&nbsp; "
        f"CPU cores: **{vm['cores']}** &nbsp;|&nbsp; "
        f"CPU capacity: **{vm['capacity']:,} MHz**"
    )
    try:
        model = get_model(resource, horizon)
    except Exception as e:
        return f"**Failed to load model from Hugging Face Model Hub.**\n\n```\n{e}\n```", usage_md, ""

    row = build_feature_row(vm)
    X = pd.DataFrame([row])[AVAILABLE_FEATURES]
    prediction = float(model.predict(X)[0])
    risk_label, risk_emoji = classify_risk(prediction)
    target_name = "CPU" if resource == "cpu" else "memory"

    result_md = (
        f"### {vm_name} — Predicted {target_name} utilization ({horizon} ahead): **{prediction:.2f}%**\n\n"
        f"Risk: {risk_emoji} **{risk_label}**\n\n"
        "*Note: predicted value can legitimately differ from current usage -- that is what a forecast "
        "is for. Random Forest moderates toward patterns seen more often in training, so it can "
        "under-react to a VM at a rare, sustained extreme reading. See the Final Report's Limitations "
        "section for details.*"
    )

    metrics = get_metrics()
    combo = (metrics or {}).get("combinations", {}).get(f"{resource}_{horizon}")
    metrics_md = ""
    if combo:
        metrics_md = (
            f"RMSE (holdout): **{combo['RMSE']:.2f}** &nbsp;|&nbsp; "
            f"R²: **{combo['R2']:.3f}** &nbsp;|&nbsp; "
            f"WAPE: **{combo['WAPE_pct']:.1f}%**"
        )

    return result_md, usage_md, metrics_md


def build_watchlist_df():
    df = pd.DataFrame(WATCHLIST, columns=["VM ID", "Predicted CPU", "Predicted memory"])
    df["Predicted CPU"] = df["Predicted CPU"].map(lambda v: f"{v:.1f}%")
    df["Predicted memory"] = df["Predicted memory"].map(lambda v: f"{v:.1f}%")
    df["Risk"] = "Critical"
    return df


# =========================================================================
# UI
# =========================================================================
with gr.Blocks(title="Predictive Resource Optimization — Decision Dashboard") as demo:
    gr.Markdown("# Predictive Resource Optimization — Decision Dashboard")
    gr.Markdown(
        "AI-Driven Predictive Resource Optimization for On-Premises Virtualized Enterprise Data "
        "Centers (QM640 Capstone)"
    )
    gr.Markdown(
        "**Data note:** KPI cards, risk tiers, RMSE chart, and detection chart use the real metrics "
        "reported in the Final Report (Random Forest vs. baselines on the actual Bitbrains-derived "
        "evaluation). The **Live Prediction** panel is different: it downloads your actual trained "
        "Random Forest model from the Hugging Face Model Hub and runs genuine inference, right here "
        "on this Space."
    )

    with gr.Row():
        gr.Markdown("**VMs scored**\n\n## 497")
        gr.Markdown("**Best model**\n\n## Random Forest")
        gr.Markdown("**RMSE vs baseline**\n\n## -10 to -41%")
        gr.Markdown("**Critical VMs now**\n\n## 36")

    gr.Markdown("#### VM risk tiers — 30-minute forecast")
    gr.Plot(value=make_risk_tier_fig(), show_label=False)

    gr.Markdown("#### Model RMSE by forecast horizon — CPU")
    gr.Plot(value=make_rmse_fig(), show_label=False)

    gr.Markdown("#### Bottleneck detection: CPU vs memory (Random Forest, 80% threshold)")
    gr.Plot(value=make_detection_fig(), show_label=False)

    gr.Markdown("---")
    gr.Markdown("### Live prediction — real model, downloaded from the Hugging Face Model Hub")

    with gr.Row():
        resource_dd = gr.Dropdown(RESOURCES, value=RESOURCES[0], label="Resource")
        horizon_dd = gr.Dropdown(HORIZONS, value=HORIZONS[0], label="Forecast horizon")
        vm_dd = gr.Dropdown(list(VM_PROFILES.keys()), value=list(VM_PROFILES.keys())[0], label="Select VM")

    usage_display = gr.Markdown()
    predict_btn = gr.Button("Predict", variant="primary")
    result_display = gr.Markdown()
    metrics_display = gr.Markdown()

    predict_btn.click(
        predict,
        inputs=[resource_dd, horizon_dd, vm_dd],
        outputs=[result_display, usage_display, metrics_display],
    )

    gr.Markdown("---")
    gr.Markdown("#### Top watchlist — highest predicted CPU, next 30 minutes")
    gr.Dataframe(value=build_watchlist_df(), interactive=False)

    gr.Markdown(
        "Source: QM640 Final Report (Results section, Tables 5 and 9) and the executed capstone "
        "notebook's decision-support output. Live Prediction calls your real model and runs genuine "
        "inference on this Space."
    )

if __name__ == "__main__":
    demo.launch()
