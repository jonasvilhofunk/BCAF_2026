# BCAF — Bidirectional Cross-Attention Fusion

[![arXiv](https://img.shields.io/badge/arXiv-2603.13941-b31b1b.svg)](https://arxiv.org/abs/2603.13941)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="https://arxiv.org/html/2603.13941v1/x1.png" alt="BCAF Architecture Overview (Figure 1)" width="800"/>
</p>

---

## Abstract

Growing waste streams and the transition to a circular economy require efficient automated waste sorting. We present **Bidirectional Cross-Attention Fusion (BCAF)**, which aligns high-resolution RGB with low-resolution HSI at their native grids via localized, bidirectional cross-attention, avoiding pre-upsampling or early spectral collapse. On the SpectralWaste dataset, BCAF achieves state-of-the-art performance of **76.4% mIoU** at 31 images/s and **75.4% mIoU** at 55 images/s.

---

## Repository Structure

```
BCAF_2026/
├── README.md
├── requirements.txt
├── .gitignore
├── ModelConfigs/
│   ├── Fusion/
│   │   ├── BCAF_RGB_1024_HSI_5.yaml
│   │   ├── BCAF_RGB_512_HSI_5.yaml
│   │   ├── BCAF_RGB_256_HSI_5.yaml
│   │   └── logitfusion_RGB_1024_HSI_5.yaml
│   ├── HSI/
│   │   ├── adapted_swin_t_HSI_3.yaml
│   │   ├── adapted_swin_t_HSI_5.yaml
│   │   ├── adapted_swin_t_HSI_7.yaml
│   │   ├── adapted_swin_t_HSI_10.yaml
│   │   └── swin_t_HSI_1.yaml
│   └── RGB/
│       ├── swin_t_rgb_256.yaml
│       ├── swin_t_rgb_512.yaml
│       ├── swin_t_rgb_1024.yaml
│       └── swin_t_rgb_2048.yaml
├── datasets/
│   └── spectralwaste_npz/
│       ├── hsi_stats.yaml
│       └── images/          ← place preprocessed .npz files here
│           ├── train/
│           ├── val/
│           └── test/
├── models/
│   └── paper/               ← place downloaded model checkpoints here
├── pyfunctions/
│   ├── augmentations.py
│   ├── backbones.py
│   ├── build_model.py
│   ├── cross_attention_fusion.py
│   ├── dataload.py
│   ├── finetune.py
│   ├── losses.py
│   ├── metrics.py
│   ├── segmentation_head.py
│   ├── wandb_image_visualization.py
│   └── warmup_scheduler.py
└── scripts/
    ├── Preprocess.ipynb
    ├── Training.ipynb
    └── Inference.ipynb
```

---

## Scripts (`scripts/`)

| Notebook | Description |
|---|---|
| `Preprocess.ipynb` | Converts the raw SpectralWaste dataset into co-registered `.npz` files (containing `rgb` and `hsi` arrays) and places them into `datasets/spectralwaste_npz/images/{train,val,test}/`. |
| `Training.ipynb` | End-to-end training pipeline: loads a YAML config, builds the model (unimodal or fusion), trains with CE+Dice loss, polynomial LR schedule with warmup, AMP, gradient accumulation, and optional W&B logging. |
| `Inference.ipynb` | Loads a trained checkpoint, runs inference on the test split, computes mIoU / per-class IoU, and visualises predictions side-by-side with ground truth. |

---

## Python Modules (`pyfunctions/`)

| Module | Description |
|---|---|
| `augmentations.py` | Configurable augmentation pipeline for RGB and/or HSI: resize, random crop, flip, rotation, scale, RGB color jitter, HSI spectral jitter, ImageNet normalization, HSI min-max / standardize normalization, and spectral channel padding. |
| `backbones.py` | Swin Transformer backbone implementations: standard Swin-T for RGB with window attention, shifted windows, patch merging; and an HSI-adapted Swin with 3D patch tokenization (`HSIPatchEmbed3D`) and spectral self-attention that preserves spectral structure. Includes `build_swin_transformer()` factory. |
| `build_model.py` | Model assembly: `UnimodalSegmentationModel` (backbone + UNet head), `LateLogitFusionModel` (two unimodal branches fused at logit level), and `CrossAttentionFusionModel` (BCAF — two backbones with bidirectional cross-attention at configurable stages + fused feature decoder). Handles checkpoint loading and `timm` pretrained weight integration. |
| `cross_attention_fusion.py` | Core BCAF module: `BidirectionalCrossAttention` and `BidirectionalCrossAttentionBlock`. Implements NxN parent–child pixel mapping between fine-grid RGB and coarse-grid HSI via `pixel_unshuffle`/`pixel_shuffle`, per-pixel multi-head cross-attention in both directions (RGB→HSI and HSI→RGB), SpectralSE collapse, and gated fusion with a learnable per-channel alpha gate. |
| `dataload.py` | `SpectralWasteDataset` PyTorch Dataset for loading `.npz` files (with `rgb` and `hsi` keys) and corresponding PNG label masks. Supports `rgb`, `hsi`, and `rgb_hsi` modalities. Includes `create_datasets()` and `create_dataloaders()` factory functions. |
| `finetune.py` | `SegmentationTrainer` class: full training loop with AdamW optimizer, polynomial / cosine LR scheduling with warmup, AMP, gradient accumulation, gradient clipping, class-frequency-based loss weighting, W&B logging, checkpoint saving, and evaluation. |
| `losses.py` | `DiceLoss` (multiclass) and `SegmentationLoss` (composite CE + Dice). Also provides `calculate_class_frequencies()` and `calculate_class_weights_from_frequencies()` for frequency-inverse class weighting. |
| `metrics.py` | Confusion-matrix-based segmentation metrics: per-class IoU, precision, recall, F1, mIoU (background excluded), and overall accuracy. |
| `segmentation_head.py` | Decoder heads: `UNetHead` (2D UNet-style decoder with skip connections for RGB), `HSIUNetHead` (spectral reduction via `SpectralSE` + 2D UNet decoder for HSI), and `FusedFeatureFusionHead` (decoder for cross-attention fused multi-stage features). `SpectralSE` implements squeeze-excitation over the spectral dimension. |
| `wandb_image_visualization.py` | W&B visualization utilities: renders side-by-side RGB/HSI-pseudoRGB, ground truth, and prediction masks with the SpectralWaste color palette (7 classes). Computes per-batch IoU metrics for logging. |
| `warmup_scheduler.py` | `GradualWarmupScheduler`: linearly warms up the learning rate over a configurable number of epochs, then hands off to a downstream scheduler (e.g., `PolynomialLR`). |

---

## Getting Started

### Installation

```bash
git clone https://github.com/jonasvilhofunk/BCAF_2026.git
cd BCAF_2026
pip install -r requirements.txt
```

### Data Preparation

1. Download the **SpectralWaste** dataset from the original source ([ferpb/spectralwaste-segmentation](https://github.com/ferpb/spectralwaste-segmentation)).
2. Run `scripts/Preprocess.ipynb` to convert the raw data into `.npz` format.
3. Preprocessed files will be placed into `datasets/spectralwaste_npz/images/{train,val,test}/`.

### Model Checkpoints

Download the pretrained BCAF model weights from 🤗 Hugging Face:

👉 [**jonasvilhofunk/BCAF_2026** on Hugging Face](https://huggingface.co/jonasvilhofunk/BCAF_2026)

Place the downloaded `.pth` checkpoint files into the `models/paper/` directory.

### Training

Open `scripts/Training.ipynb` and select a config from `ModelConfigs/`. For example, to train the best BCAF variant:

```python
config_path = "ModelConfigs/Fusion/BCAF_RGB_1024_HSI_5.yaml"
```

### Inference

Open `scripts/Inference.ipynb`, point it to a checkpoint in `models/paper/`, and run evaluation on the test set.

---

## Citation

If you use this code or the BCAF model in your research, please cite:

**BCAF (this work):**
```bibtex
@article{funk2025bcaf,
  title={Bidirectional Cross-Attention Fusion of High-Res RGB and Low-Res HSI for Multimodal Automated Waste Sorting},
  author={Funk, Jonas V. and Roming, Lukas and Michel, Andreas and B{\"a}cker, Paul and Maier, Georg and L{\"a}ngle, Thomas and Klute, Markus},
  journal={arXiv preprint arXiv:2603.13941},
  year={2025}
}
```

**SpectralWaste Dataset:**
```bibtex
@inproceedings{casao2024spectralwaste,
  title={SpectralWaste Dataset: Multimodal Data for Waste Sorting Automation},
  author={Casao, Sara and Pe{\~n}a, Fernando and Sabater, Alberto and Castill{\'o}n, Rosa and Su{\'a}rez, Dar{\'\i}o and Montijano, Eduardo and Murillo, Ana C},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5852--5858},
  year={2024},
  organization={IEEE}
}
```

**timm (Swin Transformer pretrained weights):**
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
