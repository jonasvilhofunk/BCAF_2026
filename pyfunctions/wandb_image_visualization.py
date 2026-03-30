import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import get_worker_info


DEFAULT_CLASS_NAMES = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]
DEFAULT_PALETTE_RGB = {
    0: [0, 0, 0],
    1: [218, 247, 6],
    2: [51, 221, 255],
    3: [52, 50, 221],
    4: [202, 152, 195],
    5: [0, 128, 0],
    6: [255, 165, 0]
}
HSI_BAND_INDICES = [50, 100, 150]


def _get_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for class_id, rgb in DEFAULT_PALETTE_RGB.items():
        if class_id < num_classes:
            palette[class_id] = np.array(rgb) / 255.0
    return palette


def _prepare_class_names(class_names, num_classes):
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES.copy()
    if len(class_names) < num_classes:
        class_names += [f"class_{i}" for i in range(len(class_names), num_classes)]
    return class_names


def _denormalize_rgb(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return y.permute(1, 2, 0).cpu().numpy()


def _pseudo_rgb_from_hsi(hsi):
    C = hsi.shape[0]
    hsi = hsi.float()
    if C >= 3:
        idx = [min(d, C - 1) for d in HSI_BAND_INDICES]
        img = hsi[idx].permute(1, 2, 0)
    else:
        img = hsi[0].unsqueeze(-1).repeat(1, 1, 3)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.cpu().numpy()


def _colorize_mask(mask, palette):
    return palette[mask.astype(np.int64)]


def _preprocess_predictions(images, labels, predictions, num_classes):
    images = images.cpu()
    labels = labels.cpu()
    preds = predictions.cpu()

    if labels.ndim == 4 and labels.shape[1] == 1:
        labels_proc = labels[:, 0]
    else:
        labels_proc = labels

    target_h, target_w = labels_proc.shape[-2], labels_proc.shape[-1]

    if preds.ndim == 4 and preds.shape[1] == num_classes:
        pred_idx = preds.argmax(1).cpu()
    elif preds.ndim == 3:
        pred_idx = preds.long().cpu()
    else:
        return None, None, None

    if pred_idx.shape[-2:] != (target_h, target_w):
        pred_idx = F.interpolate(
            pred_idx.unsqueeze(1).float(),
            size=(target_h, target_w),
            mode='nearest'
        ).squeeze(1).long()

    if images.ndim == 4 and (images.shape[-2] != target_h or images.shape[-1] != target_w):
        images_resized = F.interpolate(
            images,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    else:
        images_resized = images

    return images_resized, labels_proc, pred_idx


def _compute_iou_per_class(labels, pred_idx, num_classes):
    labels_np = labels.numpy()
    preds_np = pred_idx.numpy()
    ious = []
    
    for c in range(num_classes):
        lab_c = (labels_np == c)
        pred_c = (preds_np == c)
        inter = np.logical_and(lab_c, pred_c).sum(axis=(1, 2))
        union = np.logical_or(lab_c, pred_c).sum(axis=(1, 2))
        sample_ious = np.where(union == 0, np.nan, inter / (union.astype(np.float64) + 1e-12))
        mean_iou = np.nanmean(sample_ious)
        ious.append(float(mean_iou) if not np.isnan(mean_iou) else float('nan'))
    
    mIoU = float(np.nanmean(ious)) if len(ious) > 0 else float('nan')
    return ious, mIoU


def _render_visualization_figure(images_resized, labels_proc, pred_idx, palette, modality, class_names, num_classes, idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    if modality == 'rgb':
        axes[0].imshow(_denormalize_rgb(images_resized[idx]))
        axes[0].set_title("RGB")
    else:
        axes[0].imshow(_pseudo_rgb_from_hsi(images_resized[idx]))
        axes[0].set_title("HSI pseudo-RGB")
    
    gt_mask = _colorize_mask(labels_proc[idx].numpy().astype(np.int64), palette)
    pred_mask = _colorize_mask(pred_idx[idx].numpy().astype(np.int64), palette)
    
    axes[1].imshow(gt_mask)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask)
    axes[2].set_title("Prediction")
    
    for ax in axes:
        ax.axis('off')
    
    patches = [plt.Rectangle((0, 0), 1, 1, fc=palette[c]) for c in range(num_classes)]
    fig.legend(
        patches,
        class_names[:num_classes],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(num_classes, 8),
        fontsize='x-small'
    )
    plt.tight_layout()
    
    return fig


def prepare_wandb_images_SpectralWaste(images, labels, predictions, num_classes=7, modality='rgb', class_names=None, max_images=4):
    if get_worker_info() is not None:
        return []

    try:
        import wandb
    except Exception:
        return []

    class_names = _prepare_class_names(class_names, num_classes)
    palette = _get_palette(num_classes)

    images_resized, labels_proc, pred_idx = _preprocess_predictions(images, labels, predictions, num_classes)
    if images_resized is None:
        return []

    out = []
    B = images_resized.shape[0]
    for i in range(min(B, max_images)):
        fig = _render_visualization_figure(images_resized, labels_proc, pred_idx, palette, modality, class_names, num_classes, i)
        out.append(wandb.Image(fig))
        plt.close(fig)
    
    return out


def prepare_wandb_logs_SpectralWaste(images, labels, predictions, num_classes=7, modality='rgb', class_names=None, max_images=4, split='val'):
    if split != 'val':
        return {}, []

    class_names = _prepare_class_names(class_names, num_classes)
    
    images_resized, labels_proc, pred_idx = _preprocess_predictions(images, labels, predictions, num_classes)
    if images_resized is None:
        return {}, []

    ious, mIoU = _compute_iou_per_class(labels_proc, pred_idx, num_classes)
    metrics = {
        'val_mIoU': mIoU,
        **{f"val_iou_{class_names[i]}": ious[i] for i in range(num_classes)}
    }

    images_list = prepare_wandb_images_SpectralWaste(images, labels, predictions, num_classes=num_classes, modality=modality, class_names=class_names, max_images=max_images)
    
    return metrics, images_list