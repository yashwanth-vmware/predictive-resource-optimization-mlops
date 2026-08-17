"""Data Preparation

Downloads the feature-engineered dataset from the Hugging Face Dataset Hub,
performs the CHRONOLOGICAL per-VM train/test split (matching the capstone
notebook's chronological_split_by_vm() function exactly), and uploads the
resulting splits back to the Dataset Hub.

Input  (HF Dataset repo): data/features/bitbrains_feature_dataset.csv
Output (local):           predictive_resource_optimization/data/processed/{train,test}.csv
Output (HF Dataset repo): data/processed/{train,test}.csv
"""

import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "Yash0204/data-mlops-infra-predictive-optimization-final")

TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.80"))
MINIMUM_ROWS_PER_VM = int(os.getenv("MINIMUM_ROWS_PER_VM", "20"))

PROCESSED_DIR = "predictive_resource_optimization/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Download the feature-engineered dataset
# -----------------------------------------------------------------------
raw_csv_path = hf_hub_download(
    repo_id=HF_DATASET_REPO_ID,
    filename="data/features/bitbrains_feature_dataset.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(raw_csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

if "vm_id" not in df.columns:
    raise ValueError(f"Expected a 'vm_id' column. Available columns: {list(df.columns)}")


# -----------------------------------------------------------------------
# Chronological, per-VM split -- identical logic to the notebook's
# chronological_split_by_vm()
# -----------------------------------------------------------------------
def chronological_split_by_vm(frame, train_fraction=0.80, minimum_rows=20):
    train_parts, test_parts, excluded_vms = [], [], []

    for vm_id, vm_frame in frame.groupby("vm_id", sort=False):
        vm_frame = vm_frame.sort_values("timestamp").reset_index(drop=True)

        if len(vm_frame) < minimum_rows:
            excluded_vms.append(vm_id)
            continue

        split_index = int(len(vm_frame) * train_fraction)
        split_index = max(1, min(split_index, len(vm_frame) - 1))

        train_parts.append(vm_frame.iloc[:split_index].copy())
        test_parts.append(vm_frame.iloc[split_index:].copy())

    if not train_parts or not test_parts:
        raise ValueError("The chronological split did not produce train and test data.")

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True), excluded_vms


train_df, test_df, excluded = chronological_split_by_vm(df, TRAIN_FRACTION, MINIMUM_ROWS_PER_VM)

print(f"Chronological split complete.")
print(f"  Train rows : {len(train_df):,}")
print(f"  Test rows  : {len(test_df):,}")
print(f"  VMs excluded (too few rows) : {len(excluded)}")

train_path = f"{PROCESSED_DIR}/train.csv"
test_path = f"{PROCESSED_DIR}/test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Saved local splits:\n- {train_path}\n- {test_path}")

# -----------------------------------------------------------------------
# Upload back to Dataset Hub -- path_in_repo="data/processed" matches
# what train.py downloads below (Section 4).
# -----------------------------------------------------------------------
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Required for uploading processed splits.")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=PROCESSED_DIR,
    repo_id=HF_DATASET_REPO_ID,
    repo_type="dataset",
    path_in_repo="data/processed",
)
print(f"Uploaded processed splits to Dataset Hub: {HF_DATASET_REPO_ID} (data/processed/)")
