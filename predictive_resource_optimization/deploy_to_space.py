"""deploy_to_space.py
======================
Pushes the deployment/ folder (Dockerfile, README.md, app.py,
requirements.txt) to a Hugging Face Space, creating the Space if it
doesn't exist yet. Same pattern as your Predictive Maintenance project's
deploy_to_space.py.

Uploading deployment/README.md on every run (not just at Space creation)
is what guarantees the Space stays configured with the Docker SDK -- that
file's YAML front-matter is the actual source of truth Hugging Face reads,
not the space_sdk argument below, which only applies the first time the
Space is created.

Env vars:
    HF_TOKEN            (required)
    HF_SPACE_ID          (required) e.g. "YOUR_HF_USERNAME/infra-predictive-optimization-mlops"
    HF_MODEL_REPO_ID     (optional) passed through so the Space knows which
                          model repo to load at runtime
"""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_SPACE_ID = (os.getenv("HF_SPACE_ID") or "").strip()
HF_MODEL_REPO_ID = (os.getenv("HF_MODEL_REPO_ID") or "").strip()

if not HF_TOKEN or not HF_SPACE_ID:
    raise ValueError("HF_TOKEN and HF_SPACE_ID must both be set.")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_SPACE_ID, repo_type="space")
    print(f"Space exists: {HF_SPACE_ID}")
except RepositoryNotFoundError:
    print(f"Space not found. Creating: {HF_SPACE_ID}")
    try:
        create_repo(
            repo_id=HF_SPACE_ID,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=HF_TOKEN,
        )
        print(f"Space created: {HF_SPACE_ID}")
    except Exception as e:
        if "402" in str(e) or "Payment Required" in str(e):
            print(
                "Your Hugging Face account needs a PRO plan to create a Docker "
                "Space (Static Spaces are free, but this app needs a Python "
                "backend to run real inference). Deploy via Streamlit Community "
                "Cloud instead -- see Section 5.6 of the notebook."
            )
            raise SystemExit(1)
        raise

# Pass the model repo id through as a Space secret/variable so app.py can
# read it at runtime without hardcoding your username.
if HF_MODEL_REPO_ID:
    try:
        api.add_space_variable(repo_id=HF_SPACE_ID, key="HF_MODEL_REPO_ID", value=HF_MODEL_REPO_ID)
    except Exception as e:
        print(f"Note: could not set HF_MODEL_REPO_ID Space variable automatically ({e}). "
              f"Set it manually in the Space's Settings > Variables and secrets.")

api.upload_folder(
    folder_path="predictive_resource_optimization/deployment",
    repo_id=HF_SPACE_ID,
    repo_type="space",
)

print(f"Deployed to Space: https://huggingface.co/spaces/{HF_SPACE_ID}")
