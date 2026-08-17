"""Data Registration (Rubric-aligned)

Registers the feature-engineered Bitbrains dataset into a Hugging Face
**Dataset repository**. Same pattern as the Predictive Maintenance project's
data_register.py.

What it does:
1) Ensures the Dataset repo exists (creates if missing)
2) Uploads the local folder: predictive_resource_optimization/data -> repo_path: data/

Expects predictive_resource_optimization/data/features/bitbrains_feature_dataset.csv
to already exist locally -- this is the exact file the capstone notebook's
Phase 5, Step 11 ("Save the full feature dataset...") already produces.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "Yash0204/data-mlops-infra-predictive-optimization-final")
REPO_TYPE = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=HF_DATASET_REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo exists: {HF_DATASET_REPO_ID}")
except RepositoryNotFoundError:
    print(f"Dataset repo not found. Creating: {HF_DATASET_REPO_ID}")
    create_repo(repo_id=HF_DATASET_REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repo created: {HF_DATASET_REPO_ID}")

api.upload_folder(
    folder_path="predictive_resource_optimization/data",
    repo_id=HF_DATASET_REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo="data",
)

print("Uploaded dataset files to the Dataset repo under /data")
