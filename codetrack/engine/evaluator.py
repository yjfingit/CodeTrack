"""Robustness evaluation: the metrics that tracking scores cannot express.

`PR / SR / NPR` say *whether the tracker still worked*.  They cannot say *whether the
error-correction machinery did anything*, which is the claim under test.  This module
adds the three diagnostics that can:

* **detection AUROC** -- can the syndrome tell a corrupted check from a healthy one?
* **localization recall@k** -- does the locator rank the corrupted tokens first?
* **recovery gain / damage** -- are the corrupted tokens actually repaired, and are the
  healthy ones left alone?

Every routine takes plain numpy arrays so it can be unit-tested without a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["auroc", "recall_at_k", "chance_level", "summarize_detection",
           "summarize_recovery"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC, computed from the rank statistic (ties averaged)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # average the ranks inside tied groups so the result is tie-safe
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Fraction of corrupted tokens that appear in the ``top-k`` highest ``scores``.

    ``scores``: ``N`` suspicion scores (higher = more suspicious); ``labels``: binary.
    A model that cannot localize scores at chance, i.e. ``k * n_pos / N``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or scores.size == 0:
        return float("nan")
    k = int(min(k, scores.size))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[top].sum() / n_pos)


def chance_level(k: int, n: int, n_pos: int) -> float:
    """Recall@k a random scorer would achieve -- the baseline every result must beat."""
    if n == 0 or n_pos == 0:
        return float("nan")
    return float(min(k, n) * n_pos / n)


def _concat(items: Sequence[np.ndarray], dtype=np.float64) -> np.ndarray:
    arrays = [np.asarray(x, dtype=dtype).ravel() for x in items if np.asarray(x).size]
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=dtype)


def summarize_detection(syndrome_scores: Sequence[np.ndarray],
                        syndrome_labels: Sequence[np.ndarray],
                        locator_scores: Sequence[np.ndarray],
                        locator_labels: Sequence[np.ndarray],
                        syndrome_clean: Optional[Sequence[np.ndarray]] = None,
                        topk: int = 5) -> Dict[str, float]:
    """Aggregate detection AUROC and localization recall over a corpus.

    ``syndrome_clean`` are syndrome values measured on *uncorrupted* frames and act as
    the negative class.  Without them the per-check label ``1[(H @ M) > 0]`` is almost
    always 1 (a check watching 32 of 256 tokens sees a 20%-corrupted frame with
    probability ~0.999) and the AUROC is undefined.
    """
    neg_s = _concat(syndrome_clean) if syndrome_clean else np.zeros(0)
    neg_y = np.zeros(neg_s.size, dtype=bool)
    syn_s = np.concatenate([_concat(syndrome_scores), neg_s])
    syn_y = np.concatenate([_concat(syndrome_labels).astype(bool), neg_y])
    loc_s = _concat(locator_scores)
    loc_y = _concat(locator_labels).astype(bool)

    out = {
        "syndrome_auroc": auroc(syn_s, syn_y) if syn_s.size else float("nan"),
        "locator_recall_at_%d" % topk: (recall_at_k(loc_s, loc_y, topk)
                                         if loc_s.size else float("nan")),
        "locator_chance_at_%d" % topk: (chance_level(topk, loc_s.size, int(loc_y.sum()))
                                        if loc_s.size else float("nan")),
    }
    return out


def summarize_recovery(e_before: Sequence[np.ndarray],
                       e_after: Sequence[np.ndarray],
                       e_clean: Sequence[np.ndarray]) -> Dict[str, float]:
    """Masked recovery statistics.

    ``e_before``: error of the corrupted tokens before correction.
    ``e_after`` : the same tokens after correction.
    ``e_clean`` : error of the *uncorrupted* tokens after correction (damage).
    """
    b = _concat(e_before)
    a = _concat(e_after)
    c = _concat(e_clean)

    if b.size == 0 or float(b.mean()) <= 0.0:
        return {
            "error_before": float(b.mean()) if b.size else float("nan"),
            "error_after": float(a.mean()) if a.size else float("nan"),
            "recovery_gain": float("nan"),
            "damage_clean": float(c.mean()) if c.size else float("nan"),
        }

    return {
        "error_before": float(b.mean()),
        "error_after": float(a.mean()),
        "recovery_gain": float((b.mean() - a.mean()) / b.mean()),
        "damage_clean": float(c.mean()) if c.size else float("nan"),
    }
