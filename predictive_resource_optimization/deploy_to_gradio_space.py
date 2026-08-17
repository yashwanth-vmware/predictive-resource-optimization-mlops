"""deploy_to_gradio_space.py
=============================
Pushes deployment_gradio/ (app.py, requirements.txt, README.md) to a
Hugging Face **Gradio**-SDK Space, creating it if it doesn't exist. Free
personal accounts can create Gradio Spaces (they default to ZeroGPU
hardware) -- unlike Docker Spaces (deploy_to_space.py), which require PRO.

Env vars:
    HF_TOKEN               (required)
    HF_GRADIO_SPACE_ID     (required) e.g. "YOUR_HF_USERNAME/infra-predictive-optimization-mlops-gradio"
    HF_MODEL_REPO_ID       (optional) passed through as a Space variable so app.py can read it at runtime
"""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_GRADIO_SPACE_ID = (os.getenv("HF_GRADIO_SPACE_ID") or "").strip()
HF_MODEL_REPO_ID = (os.getenv("HF_MODEL_REPO_ID") or "").strip()

if not HF_TOKEN or not HF_GRADIO_SPACE_ID:
    raise ValueError("HF_TOKEN and HF_GRADIO_SPACE_ID must both be set.")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_GRADIO_SPACE_ID, repo_type="space")
    print(f"Gradio Space exists: {HF_GRADIO_SPACE_ID}")
except RepositoryNotFoundError:
    print(f"Gradio Space not found. Creating: {HF_GRADIO_SPACE_ID}")
    # space_hardware must be explicit: create_repo() with no hardware
    # argument defaults to "cpu-basic", which requires PRO for Gradio/Docker
    # SDKs. Free accounts get Gradio Spaces for free specifically on
    # ZeroGPU hardware ("zero-a10g") -- that's what the web UI provisions
    # by default when you pick Gradio + Blank as a free account, so this
    # has to be requested explicitly via the API too.
    create_repo(
        repo_id=HF_GRADIO_SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=False,
        token=HF_TOKEN,
    )
    print(f"Gradio Space created: {HF_GRADIO_SPACE_ID}")

api.upload_folder(
    folder_path="predictive_resource_optimization/deployment_gradio",
    repo_id=HF_GRADIO_SPACE_ID,
    repo_type="space",
)

if HF_MODEL_REPO_ID:
    api.add_space_variable(repo_id=HF_GRADIO_SPACE_ID, key="HF_MODEL_REPO_ID", value=HF_MODEL_REPO_ID)
    print(f"Set Space variable HF_MODEL_REPO_ID -> {HF_MODEL_REPO_ID}")

print(f"Deployed to Gradio Space: https://huggingface.co/spaces/{HF_GRADIO_SPACE_ID}")
