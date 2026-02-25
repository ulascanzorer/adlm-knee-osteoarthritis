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

## Loss Functions

Each model uses a composite loss that combines a reconstruction term with one or more supervised prediction terms. The prediction terms are scaled by a weighting factor `lambda` (default `0.5`, except `ae_pain` which uses `5.0`).

### `ae_filippo` — Simple Autoencoder

| Component | Function | Notes |
|-----------|----------|-------|
| Reconstruction | `MSELoss` | Pixel-wise mean squared error between input and output |

**Total loss:**
```
loss = MSE(x_hat, x)
```

---

### `ae_pain` — Pain-Prediction Autoencoder

| Component | Function | Notes |
|-----------|----------|-------|
| Reconstruction | `MSELoss` | Pixel-wise MSE between input and reconstructed volume |
| Pain prediction | `MSELoss` | Regression on KOOS Pain Score (continuous) |

**Total loss:**
```
loss = loss_recon + lambda_pain * loss_pain      (lambda_pain = 5.0)
```

Training is split into two phases:
1. **Phase 1 (warm-up):** Encoder and decoder are frozen; only the pain-prediction head is trained (`lr=1e-3`, 5 epochs).
2. **Phase 2 (fine-tuning):** All parameters are unfrozen and trained jointly (`lr=1e-5`, 20 epochs).

---

### `ae_tabular_input` — Tabular-Input Autoencoder

| Component | Function | Notes |
|-----------|----------|-------|
| Reconstruction | `mse_loss` | Pixel-wise MSE |
| WOMAC score | `mse_loss` | Continuous regression |
| JSN grade | `cross_entropy` | 4-class classification (grades 0–3) |
| Surgery risk | `binary_cross_entropy_with_logits` | Binary classification |

**Total loss:**
```
loss = loss_recon + lambda_tabular * (loss_womac + loss_jsn + loss_surg)      (lambda_tabular = 0.5)
```

---

### `ae_tabular_input_masked` — Masked Tabular-Input Autoencoder

Same prediction heads and reconstruction term as `ae_tabular_input`, but the supervised losses are computed only over samples for which the target label is actually present (masked losses). Additionally, **tabular dropout** (`p=0.2`) is applied during training, randomly zeroing out individual tabular input features to simulate missing data.

| Component | Function | Notes |
|-----------|----------|-------|
| Reconstruction | `mse_loss` | Pixel-wise MSE (always applied) |
| WOMAC score | masked `mse_loss` | Only over samples where target is available |
| JSN grade | masked `cross_entropy` | Only over samples where target is available |
| Surgery risk | masked `binary_cross_entropy_with_logits` | Only over samples where target is available |

**Total loss:**
```
loss = loss_recon + lambda_tabular * (loss_womac + loss_jsn + loss_surg)      (lambda_tabular = 0.5)
```