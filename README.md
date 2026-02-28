# adlm-knee-osteoarthritis
This is the repository of the Knee Osteoarthritis team in the context of the Applied Deep Learning in Medicine practical of TUM in Winter semester 2025 - 2026.
## Setup
Please run `uv sync` in order to install all used dependencies in this project. Then, you can either run python scripts using `uv run script.py` (which uses the virtual environment created by uv) or you can select the python interpreter as the one in the venv in VS code and simply run the script the normal way.
If you want to add or remove a dependency, you can do `uv add dependency` or `uv remove dependency`.
## Cluster Training
To train the models on the cluster, users can utilize and modify the `.slurm` files found in the corresponding training directories (e.g. `ae_filippo/`, `ae_tabular_input/`, `ae_pain/`, `ae_tabular_input_masked/`). These scripts are configured for SLURM workload manager and should be adjusted according to your specific cluster environment and resource requirements.

## Repository Structure
This repository contains various autoencoder implementations and inference pipelines for knee osteoarthritis analysis using MRI imaging and clinical data.
### Training Directories
- **`ae_filippo/`** - Simple Autoencoder architecture  for 3D imaging 
- **`ae_pain/`** - 3D Autoencoder  implementation specifically for pain-related prediction, using a pain prediction head in the latent space
- **`ae_tabular_input/`** - 3D Autoencoder architecture that integrates tabular clinical data with imaging, designed for multimodal input modeling.
- **`ae_tabular_input_masked/`** - Variant of tabular input autoencoder with masking support to support missing data usage during training.
### Inference Directories
- **`inference/`** - Inference pipeline utilities with scoring, clustering, visualization, and analysis functions for model predictions.
- **`tabular_input_inference/`** - Inference code for hybrid models that process both imaging and tabular inputs for predictions.
- **`tabular_input_inference_masked/`** - Inference implementation for masked tabular input models, supporting incomplete or masked clinical data.
### Data and Results
- **`csv/`** - Data folder holding CSV files including cleaned clinical data and KL osteoarthritis severity clusters.
- **`results/`** - Main output directory storing trained model artifacts, MRI results, reconstructions, and comparative analysis results.
- **`recon_samples_20/`** - Sample reconstruction outputs from autoencoder models with 20 iterations/epochs displayed as PNG images.
- **`recon_samples_50/`** - Sample reconstruction outputs from autoencoder models with 50 iterations/epochs displayed as PNG images.
### Documentation
- **`architecture_mermaid_files/`** - Contains Mermaid diagram files documenting the architecture designs of various autoencoder models.