---
title: Infra Predictive Optimization MLOps (Gradio)
emoji: 🧮
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Predictive Resource Optimization — Decision Dashboard (Gradio)

Forecasts CPU and memory utilization 5-30 minutes ahead for on-premises virtualized data center VMs,
using a Random Forest model trained on the Bitbrains GWA-T-12 FastStorage telemetry trace (QM640
capstone). Runs as a real server-side Python app; model files are fetched from the Hugging Face Model
Hub via `huggingface_hub` and inference runs on this Space.
