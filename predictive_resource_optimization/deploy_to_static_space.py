"""deploy_to_static_space.py
=============================
Pushes deployment_static/ (index.html, README.md) to a Hugging Face
**Static** Space, creating it if it doesn't exist. Unlike deploy_to_space.py
(Docker SDK), Static Spaces stay free for every account -- no PRO required --
because Gradio-Lite runs Python client-side via Pyodide instead of on an
HF-hosted server.

Uploading README.md on every run keeps the Space's YAML front-matter
(sdk: static) as the source of truth, same reasoning as the Docker path.

Env vars:
    HF_TOKEN               (required)
    HF_STATIC_SPACE_ID     (required) e.g. "YOUR_HF_USERNAME/infra-predictive-optimization-mlops-lite"
"""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_STATIC_SPACE_ID = (os.getenv("HF_STATIC_SPACE_ID") or "").strip()

if not HF_TOKEN or not HF_STATIC_SPACE_ID:
    raise ValueError("HF_TOKEN and HF_STATIC_SPACE_ID must both be set.")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_STATIC_SPACE_ID, repo_type="space")
    print(f"Static Space exists: {HF_STATIC_SPACE_ID}")
except RepositoryNotFoundError:
    print(f"Static Space not found. Creating: {HF_STATIC_SPACE_ID}")
    create_repo(
        repo_id=HF_STATIC_SPACE_ID,
        repo_type="space",
        space_sdk="static",
        private=False,
        token=HF_TOKEN,
    )
    print(f"Static Space created: {HF_STATIC_SPACE_ID}")

api.upload_folder(
    folder_path="predictive_resource_optimization/deployment_static",
    repo_id=HF_STATIC_SPACE_ID,
    repo_type="space",
)

print(f"Deployed to Static Space: https://huggingface.co/spaces/{HF_STATIC_SPACE_ID}")
