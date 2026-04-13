import numpy as np


def calculate_confusion_matrix(y_pred, y_true, num_classes):
    """
    Confusion matrix for semantic segmentation.

    Args:
        y_pred: ndarray [N,H,W] predicted class indices
        y_true: ndarray [N,H,W] ground truth class indices
        num_classes: int, including background
    Returns:
        ndarray [num_classes, num_classes] where rows=true, cols=pred
    """
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)

    mask = (y_true >= 0) & (y_true < num_classes)
    hist = np.bincount(
        num_classes * y_true[mask] + y_pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def calculate_metrics(y_pred, y_true, num_classes):
    """
    Standard segmentation metrics with background excluded from macro scores.

    Args:
        y_pred: ndarray [N,H,W] predicted class indices
        y_true: ndarray [N,H,W] ground truth class indices
        num_classes: int, including background (assumed class 0)
    Returns:
        dict: miou, accuracy, f1, class_iou, class_f1, class_precision, class_recall, support
    """
    hist = calculate_confusion_matrix(y_pred, y_true, num_classes)

    tp = np.diag(hist).astype(float)
    fp = hist.sum(0).astype(float) - tp
    fn = hist.sum(1).astype(float) - tp
    denom = tp + fp + fn
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)

    total = hist.sum()
    acc = float(tp.sum() / total) if total > 0 else 0.0

    valid = np.arange(1, num_classes) if num_classes > 1 else np.arange(num_classes)
    miou = float(np.nanmean(iou[valid])) if valid.size > 0 else 0.0
    f1_macro = float(np.nanmean(f1[valid])) if valid.size > 0 else 0.0

    return {
        'miou': miou,
        'accuracy': acc,
        'f1': f1_macro,
        'class_iou': iou.tolist(),
        'class_f1': f1.tolist(),
        'class_precision': precision.tolist(),
        'class_recall': recall.tolist(),
        'support': hist.sum(1).astype(int).tolist(),
    }