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

## How the Tabular Input Model Works

`ae_tabular_input` (and its masked variant `ae_tabular_input_masked`) is a multimodal autoencoder that processes both a 3D MRI volume and a small set of clinical tabular features simultaneously. The two streams are fused inside the latent space, and three clinical outcome heads are attached to the fused representation.

### Inputs

| Stream | Content | Normalization |
|--------|---------|---------------|
| MRI volume `x` | 3D grayscale MRI, shape `(B, 1, D, H, W)` | Rescaled to `[−1, 1]` |
| Tabular features `tab_x` | Age (`V00AGE`), abdominal circumference (`V00ABCIRC`), sex (`P02SEX`) | Age÷100, Circ÷200, Sex→0/1 |

### Architecture

The model can be broken down into five stages:

#### 1. MRI Encoder
A 4-layer 3D convolutional network (identical to the base `Autoencoder3D` encoder) that progressively downsamples the volume by a factor of 16 in each spatial dimension using stride-2 convolutions:

```
Conv3d(1→32, stride=2)  →  Conv3d(32→64, stride=2)
→  Conv3d(64→128, stride=2)  →  Conv3d(128→64, stride=2)
```

Each layer is followed by `InstanceNorm3d` and `ReLU`. Output: `(B, 64, D/16, H/16, W/16)`.

#### 2. Tabular Encoder
A two-layer MLP that compresses `T=3` clinical features into a compact 4-dimensional tabular embedding:

```
Linear(T → 32)  →  ReLU  →  Linear(32 → 4)
```

Weights are initialized with Kaiming normal. In the masked variant (`ae_tabular_input_masked`) the tabular encoder receives both the feature values and a binary missingness mask concatenated together — `Linear(2T → 32) → ReLU → Linear(32 → 4)` — so the network can learn to ignore missing entries.

#### 3. Feature Fusion
The 4-channel tabular embedding is broadcast spatially to match the MRI feature map dimensions `(D/16, H/16, W/16)`, then concatenated channel-wise with the 64-channel MRI features, giving `68` channels. A 1×1×1 convolution projects this back to `64` channels:

```
encoded_tab  →  expand to (B, 4, D', H', W')
concat([encoded_mri (B,64,...), encoded_tab (B,4,...)])  →  (B, 68, D', H', W')
Conv3d(68 → 64, kernel_size=1)  →  z  (B, 64, D', H', W')
```

This is the fused latent representation `z`.

#### 4. Decoder
The standard 4-layer 3D transposed convolutional decoder (shared with `ae_filippo`) reconstructs the full MRI volume from `z`, upsampling by 16× back to the original spatial resolution. Output activation is `Tanh`, matching the `[−1, 1]` input range.

#### 5. Prediction Heads
Three lightweight heads read from the fused latent `z` via global average pooling followed by a linear layer:

| Head | Output | Task |
|------|--------|------|
| `WOMACHead` | scalar | Regression — WOMAC total score (normalized to `[0, 1]`) |
| `JSNHead` | 4-class logits | Classification — Joint Space Narrowing grade (0–3) |
| `SurgeryHead` | scalar logit | Binary classification — knee replacement surgery |

Each head: `AdaptiveAvgPool3d(1) → flatten → Linear`.

### Forward Pass Summary

```
MRI (B,1,D,H,W)     ──► MRI Encoder ──────────────────────────────┐
                                                                   ▼
Tab (B,T)            ──► Tabular Encoder ──► expand & concat ──► z (B,64,D',H',W')
                                                                   │
                         ┌─────────────────────────────────────────┤
                         ▼                   ▼              ▼      ▼
                    Decoder            WOMACHead       JSNHead  SurgeryHead
                         │                   │              │      │
                    x_hat (recon)     womac_pred    jsn_pred  surg_pred
```

### `ae_tabular_input` vs `ae_tabular_input_masked`

| Aspect | `ae_tabular_input` | `ae_tabular_input_masked` |
|--------|--------------------|---------------------------|
| Tabular encoder input | `tab_x` only (`T` features) | `[tab_x, tab_mask]` concatenated (`2T` features) |
| Missing tabular features | Filled with `0.0` | Filled with `0.0`; mask `0` signals absence to the network |
| Prediction losses | Standard MSE / CE / BCE | **Masked** variants — loss computed only over samples with valid labels |
| Tabular dropout (training) | None | Random feature dropout with `p=0.2` to simulate missing data |
| Patient filtering | Requires all targets and all inputs present | Accepts patients with any missing targets or inputs |

The architecture diagram is available in [`architecture_mermaid_files/autoencoder_with_tabular_input/`](architecture_mermaid_files/autoencoder_with_tabular_input/).

---

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