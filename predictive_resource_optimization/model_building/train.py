"""train.py — Random Forest, All Resource/Horizon Combinations
=================================================================
Trains the Random Forest models (CPU and memory, 5/15/30-minute horizons --
6 models total) using the SAME hyperparameters as your notebook's Phase 8:
    n_estimators=80, max_depth=18, min_samples_leaf=3, max_features='sqrt'

Logs params/metrics to MLflow (rubric evidence, same as your Predictive
Maintenance project), saves models + metrics locally to model_artifacts/,
and uploads model_artifacts/ to the Hugging Face Model Hub.

Expected env vars (CI/CD friendly):
    HF_TOKEN            (required)
    HF_DATASET_REPO_ID  (optional; defaults below)
    HF_MODEL_REPO_ID    (optional; defaults below)

Optional:
    MLFLOW_EXPERIMENT (default: Predictive_Resource_Optimization)
"""

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------
# Config (CI/CD + Colab friendly, same pattern as your PM project)
# -----------------------------------------------------------------------
try:
    from google.colab import userdata  # type: ignore
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or os.getenv("HF_TOKEN", "")
except Exception:
    pass

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

DEFAULT_DATASET_REPO = "Yash0204/data-mlops-infra-predictive-optimization-final"
DEFAULT_MODEL_REPO = "Yash0204/mlops-infra-predictive-optimization-final-rf"

HF_DATASET_REPO_ID = (os.getenv("HF_DATASET_REPO_ID") or DEFAULT_DATASET_REPO).strip()
HF_MODEL_REPO_ID = (os.getenv("HF_MODEL_REPO_ID") or DEFAULT_MODEL_REPO).strip()
MLFLOW_EXPERIMENT = (os.getenv("MLFLOW_EXPERIMENT") or "Predictive_Resource_Optimization").strip()

print("Using HF_DATASET_REPO_ID:", HF_DATASET_REPO_ID)
print("Using HF_MODEL_REPO_ID  :", HF_MODEL_REPO_ID)

# -----------------------------------------------------------------------
# Real hyperparameters from your notebook's Phase 8 configuration cell --
# update these here if you retune in the notebook, so this script always
# matches what you report in the Final Report.
# -----------------------------------------------------------------------
RF_RANDOM_STATE = 42
RF_MAX_TRAIN_ROWS = 300_000
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 3
RF_MAX_FEATURES = "sqrt"
RF_N_JOBS = -1

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


def safe_mape(y_true, y_pred, epsilon=1.0):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= epsilon
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float("nan") if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom * 100)


def proportional_vm_sample(frame, maximum_rows, random_state=42):
    if len(frame) <= maximum_rows:
        return frame
    fraction = maximum_rows / len(frame)
    return (
        frame.groupby("vm_id", group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=random_state))
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------
# Load train/test from Dataset Hub
# -----------------------------------------------------------------------
download_token = HF_TOKEN if HF_TOKEN else None
train_path = hf_hub_download(
    repo_id=HF_DATASET_REPO_ID, filename="data/processed/train.csv",
    repo_type="dataset", token=download_token,
)
test_path = hf_hub_download(
    repo_id=HF_DATASET_REPO_ID, filename="data/processed/test.csv",
    repo_type="dataset", token=download_token,
)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

missing_features = [f for f in AVAILABLE_FEATURES if f not in train_df.columns]
if missing_features:
    raise ValueError(f"Missing expected feature columns: {missing_features}")

# -----------------------------------------------------------------------
# Train all 6 models, log to MLflow, save artifacts
# -----------------------------------------------------------------------
mlflow.set_experiment(MLFLOW_EXPERIMENT)

artifact_dir = Path("model_artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)
all_metrics = {}

for resource in RESOURCES:
    for horizon in HORIZONS:
        target_col = f"target_{resource}_{horizon}"
        if target_col not in train_df.columns:
            print(f"Skipping {resource}-{horizon}: target column '{target_col}' not found.")
            continue

        combo_key = f"{resource}_{horizon}"
        print(f"\n{'=' * 60}\nTraining Random Forest: {combo_key}\n{'=' * 60}")

        train_subset = train_df.dropna(subset=AVAILABLE_FEATURES + [target_col])
        test_subset = test_df.dropna(subset=AVAILABLE_FEATURES + [target_col])

        sampled_train = proportional_vm_sample(
            train_subset, RF_MAX_TRAIN_ROWS, random_state=RF_RANDOM_STATE
        ).sort_values("timestamp")

        X_train = sampled_train[AVAILABLE_FEATURES]
        y_train = sampled_train[target_col]
        X_test = test_subset[AVAILABLE_FEATURES]
        y_test = test_subset[target_col]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                max_features=RF_MAX_FEATURES,
                random_state=RF_RANDOM_STATE,
                n_jobs=RF_N_JOBS,
                bootstrap=True,
            )),
        ])

        with mlflow.start_run(run_name=f"RandomForest_{combo_key}"):
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("resource", resource)
            mlflow.set_tag("horizon", horizon)
            mlflow.log_params({
                "n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF, "max_features": RF_MAX_FEATURES,
                "train_rows": len(X_train),
            })

            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            training_seconds = time.perf_counter() - t0

            predictions = pipeline.predict(X_test)
            metrics = {
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
                "R2": float(r2_score(y_test, predictions)),
                "MAPE_pct": safe_mape(y_test, predictions),
                "WAPE_pct": wape(y_test, predictions),
                "training_seconds": round(training_seconds, 2),
            }
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["numpy.dtype"])

        print(f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}  "
              f"({training_seconds:.1f}s, {len(X_train):,} training rows)")

        model_filename = f"random_forest_{combo_key}.joblib"
        joblib.dump(pipeline, artifact_dir / model_filename)
        all_metrics[combo_key] = {
            "resource": resource, "horizon": horizon,
            "model_filename": model_filename, **metrics,
        }

metrics_path = artifact_dir / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump({
        "final_deployed_model": "RandomForest",
        "features": AVAILABLE_FEATURES,
        "combinations": all_metrics,
    }, f, indent=2)

print(f"\nSaved {len(all_metrics)} models + metrics.json to {artifact_dir.resolve()}")

# -----------------------------------------------------------------------
# Upload to HF Model Hub
# -----------------------------------------------------------------------
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set (required to upload to Model Hub). "
        "In Colab: add HF_TOKEN in Secrets. In GitHub Actions: store HF_TOKEN as a repo secret."
    )

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(artifact_dir), repo_id=HF_MODEL_REPO_ID,
        repo_type="model", path_in_repo="",
    )
except HfHubHTTPError as e:
    raise RuntimeError(f"Upload to HF Model Hub failed: {e}") from e

print(f"Uploaded model artifacts to HF Model Hub: {HF_MODEL_REPO_ID}")
