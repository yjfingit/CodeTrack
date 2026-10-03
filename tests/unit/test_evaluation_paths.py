"""Regression tests for the evaluation paths used by the validation matrix.

Two defects are locked here:

* ``Trainer._collect_diagnostics`` indexed ``out["corrupted_rgb"]`` unconditionally.  With
  image-level corruption and no token corruption the model is called with
  ``corruption=None``, so that key does not exist and the ``KeyError`` propagated into
  ``evaluate``'s ``except Exception`` handler -- every RGB low-light / occlusion / TIR
  crossover condition silently reported zero evaluated sequences.
* ``tools/summarize_paired_conditions.py`` produces the numbers the validation split is
  judged on, so its paired statistics and Holm correction are tested on synthetic data
  instead of being trusted because they look plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from codetrack.engine.losses import CodeTrackLoss  # noqa: E402
from codetrack.engine.trainer import Trainer  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from summarize_paired_conditions import (difference_in_differences, holm,  # noqa: E402
                                         paired_stats)

N, M, D = 16, 4, 8


class _StubModel:
    """Just enough model for ``load_checkpoint``: a config and a state-dict sink."""

    def __init__(self, decoder_mode: str) -> None:
        self.cfg = {"model": {"decoder_mode": decoder_mode}}
        self.loaded = None

    def load_state_dict(self, state, strict: bool = False):
        self.loaded = state
        return [], []


def test_load_checkpoint_refuses_a_decoder_mode_mismatch(tmp_path):
    path = tmp_path / "arm.pth"
    torch.save({"model": {"decoder.weight": torch.zeros(1)},
                "config": {"model": {"decoder_mode": "off"}}}, path)

    # Loading an "off" arm into a "bp" model used to leave 16 message-passing tensors at
    # their random initialisation, because strict=False only reports them in a list that
    # nobody read.
    with pytest.raises(RuntimeError, match="decoder_mode"):
        load_checkpoint(path, _StubModel("bp"), map_location="cpu")

    report = load_checkpoint(path, _StubModel("off"), map_location="cpu")
    assert report["epoch"] == 0


def test_load_checkpoint_tolerates_a_checkpoint_without_a_config(tmp_path):
    path = tmp_path / "bare.pth"
    torch.save({"model": {"w": torch.zeros(1)}, "epoch": 3}, path)

    report = load_checkpoint(path, _StubModel("bp"), map_location="cpu")

    assert report["epoch"] == 3


def _diagnostic_outputs(with_corrupted: bool) -> dict:
    torch.manual_seed(0)
    mask = torch.zeros(1, N)
    mask[0, :4] = 1.0
    mask_tir = torch.zeros(1, N)
    mask_tir[0, 8:10] = 1.0
    out = {
        "token_mask_rgb": mask,
        "token_mask_tir": mask_tir,
        "syndrome": torch.rand(1, 1, M),
        "H": torch.rand(M, N),
        "H_support": (torch.rand(M, N) > 0.5).float(),
        "locator_scattered": torch.rand(1, N),
        "corrected_rgb": torch.randn(1, N, D),
        "corrected_tir": torch.randn(1, N, D),
        "clean_tokens": {"rgb": torch.randn(1, N, D), "tir": torch.randn(1, N, D)},
    }
    if with_corrupted:
        out["corrupted_rgb"] = torch.randn(1, N, D)
        out["corrupted_tir"] = torch.randn(1, N, D)
    return out


def _stub_trainer() -> SimpleNamespace:
    return SimpleNamespace(loss_fn=CodeTrackLoss(num_parity=M, dim=D, code_dim=D))


def test_collect_diagnostics_tolerates_a_missing_corrupted_token_view():
    diag = {"syndrome": [], "syndrome_y": [], "syndrome_density": [], "syndrome_clean": [],
            "locator": [], "locator_y": [], "e_before": [], "e_after": [], "e_clean": [],
            "rec_e_before": [], "rec_e_after": [], "rec_hinge": [], "rec_active": []}

    # No corrupted_* keys: this is the image-level corruption path that used to raise.
    Trainer._collect_diagnostics(_stub_trainer(), _diagnostic_outputs(False), diag)

    assert diag["e_before"] == [] and diag["e_after"] == []
    assert len(diag["syndrome"]) == 1
    # "before" is genuinely undefined without a token-level corruption view, so it stays 0
    # and the evaluator reports recovery_gain as NaN.  "after" is still the distance from
    # the degraded tokens to their own (degraded) teacher, which is why the image-level
    # conditions must not be read as feature-recovery evidence.
    assert diag["rec_e_before"] == [0.0]
    assert diag["rec_hinge"] == [0.0]
    assert diag["rec_e_after"][0] > 0.0


def test_collect_diagnostics_still_measures_recovery_with_a_corrupted_view():
    diag = {"syndrome": [], "syndrome_y": [], "syndrome_density": [], "syndrome_clean": [],
            "locator": [], "locator_y": [], "e_before": [], "e_after": [], "e_clean": [],
            "rec_e_before": [], "rec_e_after": [], "rec_hinge": [], "rec_active": []}

    Trainer._collect_diagnostics(_stub_trainer(), _diagnostic_outputs(True), diag)

    assert len(diag["e_before"]) == 2 and len(diag["e_after"]) == 2   # rgb then tir
    assert diag["e_before"][0].shape == (4,)                          # four masked RGB tokens
    assert diag["e_before"][1].shape == (2,)                          # two masked TIR tokens
    assert diag["rec_e_before"][0] > 0.0


def test_paired_stats_reports_the_drop_and_its_direction():
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.4, 0.8, size=40)
    # a real but noisy degradation: a constant shift would have zero difference variance
    condition = reference - 0.05 + rng.normal(0.0, 0.01, size=40)

    stats = paired_stats(reference, condition, rng, draws=2000)

    assert stats["n_sequences"] == 40
    assert stats["mean_delta"] < 0.0
    assert stats["relative_drop_percent_mean"] > 0.0
    low, high = stats["relative_drop_percent_ci95"]
    assert low > 0.0 and high > 0.0
    assert stats["paired_t_p"] < 0.01
    assert stats["ci_excludes_zero"] is True


def test_paired_stats_keeps_a_null_result_null():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.4, 0.8, size=40)

    stats = paired_stats(reference, reference.copy(), rng, draws=2000)

    assert stats["mean_delta"] == 0.0
    low, high = stats["relative_drop_percent_ci95"]
    assert low <= 0.0 <= high
    assert stats["ci_excludes_zero"] is False


def test_holm_correction_is_monotone_and_conservative():
    raw = [0.001, 0.02, 0.04, 0.5]

    adjusted = holm(raw)

    assert adjusted[0] == 0.004                      # 4 x the smallest p-value
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert all(a <= 1.0 for a in adjusted)
    assert adjusted == sorted(adjusted)              # original order had ascending p-values


def test_difference_in_differences_reports_which_arm_degrades_less():
    rng = np.random.default_rng(2)
    # arm A loses 0.02 SR under the corruption, arm B loses 0.10, on the same sequences
    drop_a = 0.02 + rng.normal(0.0, 0.005, size=40)
    drop_b = 0.10 + rng.normal(0.0, 0.005, size=40)

    row = difference_in_differences(drop_a, drop_b, rng, draws=2000)

    assert row["n_sequences"] == 40
    assert row["mean_difference"] < 0.0              # negative = first arm degrades less
    low, high = row["difference_ci95"]
    assert high < 0.0 and row["ci_excludes_zero"] is True
    assert row["paired_t_p"] < 0.01


def test_difference_in_differences_is_zero_for_identical_arms():
    rng = np.random.default_rng(3)
    drop = rng.uniform(0.0, 0.1, size=30)

    row = difference_in_differences(drop, drop.copy(), rng, draws=1000)

    assert row["mean_difference"] == 0.0
    assert row["ci_excludes_zero"] is False
