"""Tracking metrics: PR (precision), SR (success/AUC), NPR (normalized precision)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["success_auc", "precision_at", "normalized_precision", "summarize"]


def _box_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``pred``/``gt``: ``N x 4`` in ``x, y, w, h``."""
    px1, py1 = pred[:, 0], pred[:, 1]
    px2, py2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]

    ix1, iy1 = np.maximum(px1, gx1), np.maximum(py1, gy1)
    ix2, iy2 = np.minimum(px2, gx2), np.minimum(py2, gy2)
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return inter / np.clip(union, 1e-9, None)


def _centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)


def success_auc(pred: np.ndarray, gt: np.ndarray, thresholds: int = 21) -> float:
    """Area under the success curve (SR / AUC), thresholds 0..1."""
    if len(pred) == 0:
        return 0.0
    iou = _box_iou(pred, gt)
    ts = np.linspace(0, 1, thresholds)
    curve = [(iou >= t).mean() for t in ts]
    return float(np.trapz(curve, ts))


def precision_at(pred: np.ndarray, gt: np.ndarray, threshold: float = 20.0) -> float:
    """Precision at a centre-error threshold in pixels."""
    if len(pred) == 0:
        return 0.0
    errors = np.linalg.norm(_centers(pred) - _centers(gt), axis=1)
    return float((errors <= threshold).mean())


def normalized_precision(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.2) -> float:
    """NPR: centre error normalised by the ground-truth box size."""
    if len(pred) == 0:
        return 0.0
    errors = np.linalg.norm(_centers(pred) - _centers(gt), axis=1)
    norm = np.sqrt(np.clip(gt[:, 2] * gt[:, 3], 1e-6, None))
    return float((errors / norm <= threshold).mean())


def summarize(results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average per-sequence metrics into the standard PR / SR / NPR triple."""
    if not results:
        return {"pr": 0.0, "sr": 0.0, "npr": 0.0}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
