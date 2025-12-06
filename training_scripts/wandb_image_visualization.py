import os
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib
if matplotlib.get_backend().lower()!="agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import get_worker_info


def _denormalize_rgb(x):
    mean = torch.tensor([0.485,0.456,0.406],device=x.device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225],device=x.device).view(3,1,1)
    y = (x*std+mean).clamp(0,1)
    return y.permute(1,2,0).cpu().numpy()


def _pseudo_rgb_from_hsi(hsi):
    """
    Create a pseudo-RGB image from HSI by selecting bands 50,100,150.
    If the HSI has fewer channels than an index, the last available band is used.
    For C<3 the first band is replicated to RGB.
    """
    C = hsi.shape[0]
    hsi = hsi.float()
    if C >= 3:
        desired = [50, 100, 150]
        # clamp indices to available range
        idx = [min(d, C - 1) for d in desired]
        img = hsi[idx].permute(1, 2, 0)
    else:
        img = hsi[0].unsqueeze(-1).repeat(1, 1, 3)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.cpu().numpy()


def _colorize_mask(mask, palette):
    return palette[mask.astype(np.int64)]


def prepare_wandb_images_SpectralWaste(images, labels, predictions, num_classes=7, modality='rgb', class_names=None, max_images=4):
    """
    Create visualization for WandB logging with separate images for input, ground truth, and prediction.

    Args:
      images: Tensor [B, C, H, W] (normalized for rgb)
      labels: Tensor [B, H, W] or [B,1,H,W]
      predictions: Tensor [B, num_classes, H, W] logits or [B, H, W] class indices
      num_classes: int
      modality: 'rgb' or 'hsi'
      class_names: optional list of names
      max_images: int, limit samples per log

    Returns:
      List[wandb.Image]
    """
    # Avoid plotting from worker subprocesses
    if get_worker_info() is not None:
        return []

    # Import wandb lazily (module import-safe)
    try:
        import wandb  # noqa: F401
    except Exception:
        return []

    # Default class names/palette
    if class_names is None:
        class_names = ["background","film","basket","cardboard","video_tape","filament","bag"]
    if len(class_names)<num_classes:
        class_names += [f"class_{i}" for i in range(len(class_names), num_classes)]

    palette = np.zeros((num_classes,3),dtype=np.float32)
    # Define the known 7-class palette (normalized [0,1]); extra classes default to black
    base = {
        0:[0,0,0],1:[218,247,6],2:[51,221,255],3:[52,50,221],
        4:[202,152,195],5:[0,128,0],6:[255,165,0]
    }
    for k,v in base.items():
        if k<num_classes: palette[k]=np.array(v)/255.0

    # Ensure shapes on CPU
    images = images.cpu()
    labels = labels.cpu()
    preds = predictions.cpu()

    # Normalize labels to [B,H,W]
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels_proc = labels[:, 0]
    else:
        labels_proc = labels

    # Make label spatial size the canonical visualization size
    target_h, target_w = labels_proc.shape[-2], labels_proc.shape[-1]

    # If predictions are logits -> argmax to get class indices
    if preds.ndim == 4 and preds.shape[1] == num_classes:
        # compute class indices first (still on CPU)
        pred_idx = preds.argmax(1).cpu()
    elif preds.ndim == 3:
        pred_idx = preds.long().cpu()
    else:
        # can't compute valid predictions
        return []

    # If prediction spatial size differs from labels, resize class indices with nearest to preserve integer labels
    if pred_idx.shape[-2:] != (target_h, target_w):
        pred_idx = F.interpolate(pred_idx.unsqueeze(1).float(), size=(target_h, target_w), mode='nearest').squeeze(1).long()

    # Ensure input images are resized to label size for consistent overlay/alignment (images may be float)
    if images.ndim == 4 and (images.shape[-2] != target_h or images.shape[-1] != target_w):
        images_resized = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
    else:
        images_resized = images

    B = images_resized.shape[0]
    out = []
    for i in range(min(B,max_images)):
        fig,axes = plt.subplots(1,3,figsize=(12,4))
        if modality=='rgb':
            axes[0].imshow(_denormalize_rgb(images_resized[i])); axes[0].set_title("RGB")
        else:
            axes[0].imshow(_pseudo_rgb_from_hsi(images_resized[i])); axes[0].set_title("HSI pseudo-RGB")
        gt_mask = _colorize_mask(labels_proc[i].numpy().astype(np.int64), palette)
        pred_mask = _colorize_mask(pred_idx[i].numpy().astype(np.int64), palette)
        axes[1].imshow(gt_mask); axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask); axes[2].set_title("Prediction")
        for a in axes: a.axis('off')
        patches = [plt.Rectangle((0,0),1,1,fc=palette[c]) for c in range(num_classes)]
        fig.legend(patches,class_names[:num_classes],loc='lower center',bbox_to_anchor=(0.5,-0.02),ncol=min(num_classes,8),fontsize='x-small')
        plt.tight_layout()
        out.append(wandb.Image(fig))
        plt.close(fig)
    return out


def _compute_iou_per_class(labels, pred_idx, num_classes):
    """
    labels: Tensor [B,H,W] (cpu)
    pred_idx: Tensor [B,H,W] (cpu)
    returns: list of IoU per class (floats or np.nan), mIoU (float)
    """
    labels_np = labels.numpy()
    preds_np = pred_idx.numpy()
    ious = []
    for c in range(num_classes):
        lab_c = (labels_np == c)
        pred_c = (preds_np == c)
        inter = np.logical_and(lab_c, pred_c).sum(axis=(1,2))
        union = np.logical_or(lab_c, pred_c).sum(axis=(1,2))
        # compute IoU per sample then mean across batch, treat union==0 as nan for that sample
        sample_ious = np.where(union == 0, np.nan, inter / (union.astype(np.float64) + 1e-12))
        # mean across batch, ignoring nan
        mean_iou = np.nanmean(sample_ious)
        ious.append(float(mean_iou) if not np.isnan(mean_iou) else float('nan'))
    # mean across classes, ignoring nan
    mIoU = float(np.nanmean(ious)) if len(ious) > 0 else float('nan')
    return ious, mIoU


def prepare_wandb_logs_SpectralWaste(images, labels, predictions, num_classes=7,
                                     modality='rgb', class_names=None, max_images=4,
                                     split='val'):
    """
    Returns (metrics_dict, wandb_images_list).

    metrics_dict contains only val_mIoU and val_iou_<class_name> keys when split=='val'.
    wandb_images_list is the same list produced by prepare_wandb_images_SpectralWaste.

    If split != 'val', returns ({}, []).
    """
    # only log metrics for validation
    if split != 'val':
        return {}, []

    # replicate small part of prepare_wandb_images_SpectralWaste to get pred_idx for IoU
    # ensure shapes on CPU
    images = images.cpu()
    labels = labels.cpu()
    preds = predictions.cpu()

    # Normalize labels to [B,H,W]
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels_proc = labels[:, 0]
    else:
        labels_proc = labels

    # Make label spatial size the canonical visualization size
    target_h, target_w = labels_proc.shape[-2], labels_proc.shape[-1]

    # If predictions are logits -> argmax to get class indices
    if preds.ndim == 4 and preds.shape[1] == num_classes:
        # compute class indices first (still on CPU)
        pred_idx = preds.argmax(1).cpu()
    elif preds.ndim == 3:
        pred_idx = preds.long().cpu()
    else:
        # can't compute valid predictions
        return {}, []

    # If prediction spatial size differs from labels, resize class indices with nearest to preserve integer labels
    if pred_idx.shape[-2:] != (target_h, target_w):
        pred_idx = F.interpolate(pred_idx.unsqueeze(1).float(), size=(target_h, target_w), mode='nearest').squeeze(1).long()

    # Ensure input images are resized to label size for consistent overlay/alignment (images may be float)
    if images.ndim == 4 and (images.shape[-2] != target_h or images.shape[-1] != target_w):
        images_resized = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
    else:
        images_resized = images

    # prepare class names
    if class_names is None:
        class_names = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]
    if len(class_names) < num_classes:
        class_names += [f"class_{i}" for i in range(len(class_names), num_classes)]

    # compute IoUs
    ious, mIoU = _compute_iou_per_class(labels_proc, pred_idx, num_classes)
    metrics = {}
    metrics['val_mIoU'] = mIoU
    for i in range(num_classes):
        metrics[f"val_iou_{class_names[i]}"] = ious[i]

    # get images (this function handles worker-process and wandb import checks)
    images_list = prepare_wandb_images_SpectralWaste(images, labels, predictions,
                                                    num_classes=num_classes,
                                                    modality=modality,
                                                    class_names=class_names,
                                                    max_images=max_images)
    return metrics, images_list