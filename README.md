<h1 align="center">
  Bidirectional Cross-Attention Fusion of High-Res RGB and Low-Res HSI
  for Multimodal Automated Waste Sorting
</h1>

<p align="center">
  <sub>
  <a href="https://www.linkedin.com/in/jonas-vilho-funk-b66636158">Jonas V. Funk</a><sup>1,2</sup>
  &nbsp;&bull;&nbsp;
  Lukas Roming<sup>2</sup>
  &nbsp;&bull;&nbsp;
  Andreas Michel<sup>2</sup>
  &nbsp;&bull;&nbsp;
  Paul B&auml;cker<sup>2</sup>
  &nbsp;&bull;&nbsp;
  Georg Maier<sup>2</sup>
  &nbsp;&bull;&nbsp;
  Thomas L&auml;ngle<sup>1,2</sup>
  &nbsp;&bull;&nbsp;
  Markus Klute<sup>1</sup>
  </sub>
</p>

<p align="center">
  <sub><sup>1</sup>Karlsruhe Institute of Technology &nbsp;&nbsp;
  <sup>2</sup>Fraunhofer IOSB</sub>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.13941-b31b1b.svg)](https://arxiv.org/abs/2603.13941) [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Weights-yellow)](https://huggingface.co/jonasvilhofunk/BCAF_2026/tree/main)

</div>
    
---
&nbsp;
# Abstract

We present **Bidirectional Cross-Attention Fusion (BCAF)**, which aligns high-resolution RGB 
with low-resolution HSI at their native grids via localized, bidirectional cross-attention, 
avoiding spatial pre-upsampling or early spectral collapse. (On the [SpectralWaste](https://github.com/ferpb/spectralwaste-segmentation) dataset, BCAF achieves 
state-of-the-art performance of **76.4% mIoU** at 31 images/s.)

<p align="center">
  <img src="https://github.com/user-attachments/assets/ff90dc60-68e8-4e23-a965-827e5dbbc105"
       alt="BCAF Figure 1" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc606527-8a4a-4ec0-87e0-72eca8b9b024"
       alt="BCAF Figure 2" />
</p>

---

&nbsp;

# Getting Started

## Scripts
| Notebook | Description |
|---|---|
| `Preprocess.ipynb` | Converts the SpectralWaste dataset into `.npz` files (containing `rgb` and `hsi` arrays). |
| `Evaluate.ipynb` | Loads trained checkpoints, visualises predictions and computes mIoU on the test split. |
| `Training.ipynb` | End-to-end training pipeline: loads a YAML config, builds the model (unimodal or fusion) and trains. |

&nbsp;

## Installation

```bash
git clone https://github.com/jonasvilhofunk/BCAF_2026.git
cd BCAF_2026
python -m venv .Env_BCAF
source .Env_BCAF/bin/activate
pip install -r requirements.txt
```
> **Requirements:** Python ≥ 3.12, (CUDA-compatible GPU tested with CUDA 12.6).

&nbsp;
## Model Checkpoints

Download the pretrained weights from [**Hugging Face**](https://huggingface.co/jonasvilhofunk/BCAF_2026/tree/main).
The easiest way clones them directly into the expected location:

```bash
apt install git-lfs
git lfs install
git clone https://huggingface.co/jonasvilhofunk/BCAF_2026 models/paper/
```

Alternatively, download the `.pth` files manually and place them into `models/paper/`.

&nbsp;

## Data Preparation

This repository is evaluated on the [**SpectralWaste**](https://github.com/ferpb/spectralwaste-segmentation) dataset.

1. Download SpectralWaste from [ferpb/spectralwaste-segmentation](https://github.com/ferpb/spectralwaste-segmentation) and place the `spectralwaste_segmentation` folder into `datasets/`.
2. Open `scripts/Preprocess.ipynb` and run all cells, this converts the raw data into `.npz` files and computes HSI statistics.
   
&nbsp;

## Reproduce Results on SpectralWaste

With the weights and data in place, open `scripts/Evaluate.ipynb` and run all cells to reproduce the metrics from the paper and visualise segmentation predictions on the test split.

| Backbone | Modality | mIoU ↑ | Img./s ↑ |
|---|---|---|---|
| Swin-T | RGB-256 | 65.8 ± 1.2 | 141 |
| Swin-T | RGB-512 | 71.1 ± 0.6 | 135 |
| Swin-T | RGB-1024 | 71.6 ± 0.3 | 60 |
| Swin-T | RGB-2048 | 68.4 ± 0.8 | 15 |
| Swin-T | HSI-1 | 60.9 ± 0.2 | 141 |
| Adapted Swin-T | HSI-3 | 59.7 ± 0.7 | 114 |
| Adapted Swin-T | HSI-5 | 60.3 ± 0.9 | 119 |
| Adapted Swin-T | HSI-7 | 59.0 ± 1.5 | 91 |
| Adapted Swin-T | HSI-10 | 57.8 ± 1.2 | 68 |
| Logit Fusion | RGB-1024 + HSI-5 | 72.6 ± 0.8 | 39 |
| BCAF | RGB-256 + HSI-5 | 71.1 ± 0.4 | 54 |
| BCAF | RGB-512 + HSI-5 | 75.4 ± 0.2 | 55 |
| **BCAF** | **RGB-1024 + HSI-5** | **76.4 ± 0.4** | **31** |

> SegFormer (MiT-B0/B2) baselines and full per-class IoU are in the [paper](https://arxiv.org/abs/2603.13941).


---
&nbsp;

&nbsp;

# Custom Datasets

BCAF is evaluated on SpectralWaste for reproducibility, but the method is
**modality-agnostic** and works for any co-registered RGB + lower-resolution,
high-channel auxiliary sensor (NIR, SWIR, multispectral, thermal, etc.).

## Step 1: Prepare Your Data

Each sample must be saved as a single `.npz` file containing two arrays:

```python
np.savez(
    "sample_001.npz",
    rgb = rgb_array,   # shape (3, H_rgb, W_rgb),  dtype float32
    hsi = hsi_array,   # shape (S, H_hsi, W_hsi),  dtype float32
)
```

RGB and HSI must be **co-registered** (spatially aligned), and the spatial ratio
`r = H_rgb / H_hsi` must be a consistent integer across all samples.

Organize your files into the following layout:


```
datasets/
├── <your_dataset>/
│   ├── labels_rgb/
│   │   ├── train/          ← grayscale PNG masks (pixel value = class index)
│   │   ├── val/
│   │   └── test/
│   └── labels_hyper_lt/
│       ├── train/
│       ├── val/
│       └── test/
└── <your_dataset>_npz/
    ├── hsi_stats.yaml      ← REQUIRED (generated via Preprocess.ipynb)
    └── images/
        ├── train/          ← .npz files, each with 'rgb' and 'hsi' arrays
        ├── val/
        └── test/

```
&nbsp;

## Step 2: Compute HSI Statistics

Open `scripts/Preprocess.ipynb` and set `dataset_name`.
The notebook computes per-band mean and std over the training split and writes
`hsi_stats.yaml`, which is used for normalization during training.

&nbsp;

## Step 3: Update `pyfunctions/dataload.py`

Adjust all lines marked with `# <<< CUSTOM DATASET`:

```python
DATASET_NAME       = "<your_dataset_name>"                                             # <<< CUSTOM DATASET: must match dataset_name in your YAML config (case-insensitive)
DEFAULT_STATS_PATH = REPO_ROOT / "datasets" / "<your_dataset>_npz" / "hsi_stats.yaml"  # <<< CUSTOM DATASET: point to your hsi_stats.yaml (generated by Preprocess.ipynb)
IMAGE_ROOT         = REPO_ROOT / "datasets" / "<your_dataset>_npz"                     # <<< CUSTOM DATASET: root of your <dataset>_npz/ folder (must contain images/train|val|test/)
LABELS_ROOT        = REPO_ROOT / "datasets" / "<your_dataset>"                         # <<< CUSTOM DATASET: root of your labels folder (must contain labels_rgb/ and labels_hyper_lt/)
NUM_CLASSES = <N>                                                                      # <<< CUSTOM DATASET: total number of classes including background
CLASS_NAMES = ["background", "class_1", ...]                                           # <<< CUSTOM DATASET: one name per class, length must equal NUM_CLASSES
LABEL_SUFFIXES = ["_grayscale.png", ".png"]                                            # <<< CUSTOM DATASET: candidate label filename suffixes controls how label files are located: s `stem + suffix`
```
&nbsp;

## Step 4: Create a YAML Config

Copy `ModelConfigs/Fusion/BCAF_RGB_1024_HSI_5.yaml` (for fusion) or the appropriate RGB/HSI config and edit only the `CONFIGURE` block at the top — the `FIXED` block below it does not need to change.

**Fusion config** (`ModelConfigs/Fusion/`):
```yaml
# ═══════════════════════════════════════════════════════════
# CONFIGURE — change these to train on a new dataset / setup
# ═══════════════════════════════════════════════════════════
dataset_name: "<your_dataset_name>"   # ← must match the name used in dataload.py
modality: "rgb_hsi"
model_type: "featurefusion"

rgb_model_config_path: "ModelConfigs/RGB/<your_rgb_config>.yaml"
rgb_checkpoint_path:   "models/paper/<your_rgb_checkpoint>.pth"   # ← from Phase 1

hsi_model_config_path: "ModelConfigs/HSI/<your_hsi_config>.yaml"
hsi_checkpoint_path:   "models/paper/<your_hsi_checkpoint>.pth"   # ← from Phase 1

_rgb_size: &rgb_size <H_rgb>          # ← RGB spatial resolution
_hsi_size: &hsi_size <H_hsi>          # ← HSI spatial resolution
_in_chans: &in_chans <S_padded>       # ← HSI channels after padding (pad_channels_to)

_batch_size: &batch_size 2
_epochs:     &epochs 50
_checkpoint_dir: &checkpoint_dir "./models/checkpoints/<run_name>"
```

**RGB-only config** (`ModelConfigs/RGB/`):
```yaml
dataset_name: "<your_dataset_name>"
modality: "rgb"

_rgb_size: &rgb_size <H_rgb>          # ← drives resize, crop, and backbone img_size
_batch_size: &batch_size 2
_epochs:     &epochs 50
_checkpoint_dir: &checkpoint_dir "./models/checkpoints/<run_name>"
```

**HSI-only config** (`ModelConfigs/HSI/`):
```yaml
dataset_name: "<your_dataset_name>"
modality: "hsi"

_hsi_size: &hsi_size <H_hsi>    # ← spatial size
_in_chans: &in_chans <S_padded> # ← channels after padding (must equal pad_channels_to)
_spectral_group_size: &spectral_group_size <G>       # ← in_chans / G = number of spectral slices K
# K rule of thumb:
#   coarse tasks (paper / metal / plastic / ...):       K = 1–3
#   fine-grained spectral tasks (PET/PP/HDPE/...):   K = 5–10

_batch_size: &batch_size 2
_epochs:     &epochs 50
_checkpoint_dir: &checkpoint_dir "./models/checkpoints/<run_name>"
```
&nbsp;

## Step 5: Train

Open `scripts/Training.ipynb`. It has three ready-to-run sections — run them in order for best results.

**1. Train RGB Unimodal** (Phase 1a)

In the *Train RGB Unimodal* cell, set the config path and run:
```python
config_path = project_root / "ModelConfigs" / "RGB" / "swin_t_RGB_256.yaml"  # ← swap size (256/512/1024)
```

**2. Train HSI Unimodal** (Phase 1b)

In the *Train HSI Unimodal* cell:
```python
config_path = project_root / "ModelConfigs" / "HSI" / "adapted_swin_t_HSI_5.yaml"  # ← swap K (1/3/5/7/10)
```

**3. Train Fusion Model** (Phase 2)

In the *Train Fusion Model* cell, point the config to the checkpoints produced in Phases 1a/b
(set `rgb_checkpoint_path` and `hsi_checkpoint_path` in the YAML), then run:
```python
config_path = project_root / "ModelConfigs" / "Fusion" / "BCAF_RGB_256_HSI_5.yaml"  # ← swap config
```

> The fusion model can also be trained from random initialization, but initializing from
> unimodal checkpoints stabilizes training and yields better results.

---
&nbsp;

&nbsp;

# Citation

If you use this code or the BCAF model in your research, please cite:
```bibtex
@article{funk2026bcaf,
      title={Bidirectional Cross-Attention Fusion of High-Res RGB and Low-Res HSI for Multimodal Automated Waste Sorting}, 
      author={Jonas V. Funk and Lukas Roming and Andreas Michel and Paul Bäcker and Georg Maier and Thomas Längle and Markus Klute},
      year={2026},
      eprint={2603.13941},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.13941}, 
}

```

&nbsp;

## Acknowledgements

This work builds on [Swin Transformer](https://github.com/microsoft/Swin-Transformer),
using ImageNet-1K pretrained Swin-T weights loaded via
[timm](https://github.com/huggingface/pytorch-image-models).



## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.
