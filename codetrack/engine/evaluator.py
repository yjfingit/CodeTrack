"""Robustness evaluation: the metrics that tracking scores cannot express.

`PR / SR / NPR` say *whether the tracker still worked*.  They cannot say *whether the
error-correction machinery did anything*, which is the claim under test.  This module
adds the three diagnostics that can:

* **detection AUROC** -- can the syndrome tell a corrupted check from a healthy one?
* **density Spearman** -- does the syndrome *rank* checks by how much damage they watch?
* **localization precision@k** -- of the k most suspicious tokens, how many are corrupted
  (chance = the corruption ratio); recall@k is also reported but its ceiling is
  ``k / n_corrupted``, which is ~0.1 at 20% corruption.
* **recovery gain / damage** -- are the corrupted tokens actually repaired, and are the
  healthy ones left alone?

Every routine takes plain numpy arrays so it can be unit-tested without a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["auroc", "spearman", "recall_at_k", "precision_at_k", "chance_level",
           "summarize_detection", "summarize_recovery"]


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

    ``scores``: ``F x N`` suspicion scores (higher = more suspicious), one row per frame;
    ``labels``: ``F x N`` binary.  A 1-D input is treated as a single frame.

    Ranking is done **per frame**: a global top-k over a whole corpus would always
    surface the tokens of whichever frames happened to be worst, which measures the
    frame difficulty rather than the localizer.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    y = np.atleast_2d(np.asarray(labels)).astype(bool)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must have the same shape")

    recalls = []
    for row_s, row_y in zip(s, y):
        n_pos = int(row_y.sum())
        if n_pos == 0 or row_s.size == 0:
            continue
        kk = int(min(k, row_s.size))
        top = np.argsort(-row_s, kind="mergesort")[:kk]
        recalls.append(float(row_y[top].sum() / n_pos))
    return float(np.mean(recalls)) if recalls else float("nan")


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Fraction of the ``top-k`` highest ``scores`` that are corrupted.

    Preferred over :func:`recall_at_k` here, because at 20% corruption a check pool of
    256 tokens has ~51 corrupted ones, so recall@5 is capped at 5/51 = 0.098 no matter
    how good the model is.  Precision@5 has a ceiling of 1.0 and its chance level is
    simply the corruption ratio, which makes "0.40 vs 0.20 chance" immediately readable.
    Ranking is per frame.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    y = np.atleast_2d(np.asarray(labels)).astype(bool)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must have the same shape")

    precisions = []
    for row_s, row_y in zip(s, y):
        if row_y.sum() == 0 or row_s.size == 0:
            continue
        kk = int(min(k, row_s.size))
        top = np.argsort(-row_s, kind="mergesort")[:kk]
        precisions.append(float(row_y[top].mean()))
    return float(np.mean(precisions)) if precisions else float("nan")


def chance_level(k: int, n: int, n_pos: int) -> float:
    """Recall@k a random scorer would achieve -- the baseline every result must beat.

    With ``k`` slots drawn uniformly from ``n`` items of which ``n_pos`` are positive,
    the expected number of positives in the top-k is ``k * n_pos / n``, so the expected
    *recall* is ``(k * n_pos / n) / n_pos = k / n``.  Returning ``k * n_pos / n`` (the
    expected hit count, not the recall) is the easy mistake: it exceeds 1 as soon as
    ``k > n / n_pos``, e.g. 5 / (20% of 256) = ~1.0-1.8, which makes the baseline look
    unbeatable and the metric useless.
    """
    if n == 0 or n_pos == 0:
        return float("nan")
    return float(min(k, n) / n)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i
        while j + 1 < sorted_x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(scores: np.ndarray, targets: np.ndarray) -> float:
    """Rank correlation -- the right metric for a *graded* per-check target.

    The syndrome is now trained to regress the corruption **density** of each check, not
    a binary "did it fire".  Density is continuous, so AUROC against a thresholded label
    throws away most of the supervision signal; Spearman measures whether the checks the
    model finds worst really are the ones that watch the most damaged tokens.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    if s.size < 2 or s.std() == 0 or t.std() == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(s), _rankdata(t))[0, 1])


def _concat(items: Sequence[np.ndarray], dtype=np.float64) -> np.ndarray:
    arrays = [np.asarray(x, dtype=dtype).ravel() for x in items if np.asarray(x).size]
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=dtype)


def _stack(items: Sequence[np.ndarray], dtype=np.float64) -> np.ndarray:
    """Keep the per-frame rows: ``F x N``.  Used for per-frame ranking metrics."""
    arrays = [np.atleast_2d(np.asarray(x, dtype=dtype)) for x in items
              if np.asarray(x).size]
    return np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 0), dtype=dtype)


def summarize_detection(syndrome_scores: Sequence[np.ndarray],
                        syndrome_labels: Sequence[np.ndarray],
                        locator_scores: Sequence[np.ndarray],
                        locator_labels: Sequence[np.ndarray],
                        syndrome_clean: Optional[Sequence[np.ndarray]] = None,
                        syndrome_density: Optional[Sequence[np.ndarray]] = None,
                        density_threshold: float = 0.5,
                        topk: int = 5) -> Dict[str, float]:
    """Aggregate detection metrics and localization recall over a corpus.

    ``syndrome_labels``    per-check binary label (kept for a coarse "is this check badly
                           hit" reading; the training target itself is a density).
    ``syndrome_density``   per-check corruption **density** in ``[0, 1]`` -- the actual
                           supervision target.  Reported as Spearman + MAE, because a
                           continuous target needs a continuous metric.
    ``syndrome_clean``     syndrome values measured on *uncorrupted* frames, used as the
                           negative class of the AUROC.  They are essential: a check
                           watching 32 of 256 tokens fires on a 20%-corrupted frame with
                           probability ~0.999, so without true negatives the AUROC is
                           undefined.
    ``density_threshold``  a check counts as "badly hit" above this density, which is
                           what turns the density into an AUROC label.
    """
    neg_s = _concat(syndrome_clean) if syndrome_clean else np.zeros(0)
    neg_y = np.zeros(neg_s.size, dtype=bool)
    syn_s = np.concatenate([_concat(syndrome_scores), neg_s])
    syn_y = np.concatenate([_concat(syndrome_labels).astype(bool), neg_y])
    # localization is ranked *within* a frame, so the frame axis must survive
    loc_s = _stack(locator_scores)
    loc_y = _stack(locator_labels).astype(bool)

    n_frames, n_tokens = loc_s.shape if loc_s.size else (0, 0)
    n_corrupt = int(loc_y.sum()) if loc_y.size else 0
    ratio = (n_corrupt / (n_frames * n_tokens)) if (n_frames and n_tokens) else float("nan")
    out = {
        "syndrome_auroc": auroc(syn_s, syn_y) if syn_s.size else float("nan"),
        # precision@k is the headline localization number; recall@k is kept because its
        # ceiling is k/n_pos (~0.1 at 20% corruption), so it must be read with that in mind
        "locator_precision_at_%d" % topk: (precision_at_k(loc_s, loc_y, topk)
                                            if loc_s.size else float("nan")),
        "locator_precision_chance": ratio,
        "locator_recall_at_%d" % topk: (recall_at_k(loc_s, loc_y, topk)
                                         if loc_s.size else float("nan")),
        "locator_chance_at_%d" % topk: (chance_level(topk, n_tokens, 1)
                                        if loc_s.size else float("nan")),
    }

    if syndrome_density is not None:
        dens = _concat(syndrome_density)
        sc = _concat(syndrome_scores)
        if dens.size == sc.size and dens.size:
            out["syndrome_density_spearman"] = spearman(sc, dens)
            out["syndrome_density_mae"] = float(np.abs(sc - dens).mean())
            hard = dens > float(density_threshold)
            if hard.any() and (~hard).any():
                out["syndrome_density_auroc"] = auroc(sc, hard)
            else:
                out["syndrome_density_auroc"] = float("nan")
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
