import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks (multiclass).
    Expects logits and integer labels. No ignore_index handling.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred_logits, y_true_labels):
        # y_pred_logits: (B, C, H, W)
        # y_true_labels: (B, H, W) or (B, 1, H, W) or one-hot (B, C, H, W)
        num_classes = y_pred_logits.shape[1]
        probs = F.softmax(y_pred_logits, dim=1)  # (B, C, H, W)

        if y_true_labels.ndim == 3:
            one_hot = F.one_hot(y_true_labels, num_classes).permute(0, 3, 1, 2).float()
        elif y_true_labels.ndim == 4 and y_true_labels.shape[1] == 1:
            one_hot = F.one_hot(y_true_labels.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
        elif y_true_labels.ndim == 4 and y_true_labels.shape[1] == num_classes:
            one_hot = y_true_labels.float()
        else:
            raise ValueError(f"Bad label shape {y_true_labels.shape}")

        inter = (probs * one_hot).sum(dim=(2, 3))
        pred_sum = probs.sum(dim=(2, 3))
        true_sum = one_hot.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (pred_sum + true_sum + self.smooth)
        return 1 - dice.mean()



class SegmentationLoss(nn.Module):
    """Composite CE + Dice loss for segmentation."""
    def __init__(self, loss_type='ce_dice', num_classes=None, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if class_weights is not None and isinstance(class_weights, (list, np.ndarray)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights
        if 'ce' in self.loss_type:
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        if 'dice' in self.loss_type:
            self.dice = DiceLoss()

    def forward(self, logits, targets):
        parts = []
        if 'ce' in self.loss_type:
            if self.class_weights is not None and hasattr(self, 'ce') and self.ce.weight.device != logits.device:
                self.ce.weight = self.class_weights.to(logits.device)
            parts.append(self.ce_weight * self.ce(logits, targets))
        if 'dice' in self.loss_type:
            parts.append(self.dice_weight * self.dice(logits, targets))
        if not parts:
            raise ValueError(f"Unknown loss_type {self.loss_type}")
        total = sum(parts)
        if torch.isnan(total) or torch.isinf(total):
            raise ValueError("Loss is NaN/Inf")
        return total


def calculate_class_frequencies(loader, num_classes, label_key, device):
    counts = torch.zeros(num_classes, device=device)
    for batch in tqdm(loader, desc="Class freq"):
        labels = batch[label_key].to(device)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        counts += torch.bincount(labels.view(-1), minlength=num_classes).to(device)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, device=device) / num_classes
    return counts / total


def calculate_class_weights_from_frequencies(class_frequencies, strategy, device, epsilon=1e-6, num_classes_for_norm=None):
    if class_frequencies is None:
        return None
    if strategy == 'frequency':
        nz = class_frequencies[class_frequencies > 0]
        median = torch.median(nz) if len(nz) > 0 else torch.tensor(1.0, device=device)
        weights = median / (class_frequencies + epsilon)
    else:
        return None
    if num_classes_for_norm and weights.sum() > 0:
        weights = weights / weights.sum() * num_classes_for_norm
    return weights.to(device)