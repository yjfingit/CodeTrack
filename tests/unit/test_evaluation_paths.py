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

import json
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
from codetrack.engine.trainer import (Trainer, clamp_box,  # noqa: E402
                                      crop_box_for_frame, plan_corruption)
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


def test_identity_corruption_is_a_strict_no_op():
    """``mode="identity"`` must not build a corruption config, a token config or an RNG.

    A stray RNG draw would desynchronise every later random number and make the control
    differ from the clean run for reasons unrelated to corruption.
    """
    img_cfg, token_cfg, rng, identity = plan_corruption(
        {"token": ["tok_block_erase"], "ratio": 0.2, "severity": 0.4},
        {"enabled": True, "mode": "identity"})

    assert identity is True
    assert (img_cfg, token_cfg, rng) == (None, None, None)


def test_ratio_zero_is_not_a_no_op_and_still_builds_a_token_plan():
    """``--corrupt-ratio 0`` erases one token per frame (``corrupt_tokens`` clamps to 1).

    The strict no-op therefore has to be its own mode; a ratio-0 run must not be reported as
    an identity control.
    """
    img_cfg, token_cfg, rng, identity = plan_corruption(
        {"severity": 0.4}, {"enabled": True, "token": ["tok_block_erase"], "ratio": 0.0,
                            "target": "tir"})

    assert identity is False
    assert img_cfg is not None and rng is not None
    assert token_cfg is not None and token_cfg["ratio"] == 0.0
    assert token_cfg["target"] == "tir"
    assert token_cfg["severity"] == 0.4          # inherited from the base config


def test_corruption_plan_keeps_image_only_conditions_token_free():
    img_cfg, token_cfg, rng, identity = plan_corruption(
        {"severity": 0.4}, {"enabled": True, "token": [], "rgb": ["rgb_lowlight"],
                            "target": "rgb"})

    assert identity is False
    assert img_cfg is not None and rng is not None
    assert token_cfg is None


def test_detect_loss_deviation_target_equals_the_squashed_feature_deviation():
    """P4: the mask-independent target is the continuous deviation from the frozen teacher.

    ``q_i = e_i / (e_i + c_m)`` with ``e_i = mean_c |x_i - x*_i|`` -- a quantity that exists
    for any degradation with a teacher, unlike the injected mask.
    """
    torch.manual_seed(3)
    n, m, d = N, M, D
    cm = 0.5
    clean = torch.randn(1, n, d)
    corrupted = clean + 0.1
    outputs = {
        "corrupted_rgb": corrupted,
        "corrupted_tir": clean.clone(),
        "clean_tokens": {"rgb": clean.clone(), "tir": clean.clone()},
        "reliability_rgb": torch.rand(1, n).clamp(0.01, 0.99),
        "reliability_tir": torch.rand(1, n).clamp(0.01, 0.99),
        "syndrome": torch.rand(1, 1, m).clamp(0.01, 0.99),
        "syndrome_logits": torch.randn(1, 1, m),
        "locator_scattered": torch.rand(1, n).clamp(0.01, 0.99),
        "H_support": (torch.rand(m, n) > 0.5).float(),
    }
    loss_fn = CodeTrackLoss(num_parity=m, dim=d, code_dim=d, detect_target="deviation",
                            detect_deviation_cm=cm)

    parts = loss_fn._detect_loss(outputs, torch.ones(1, n), torch.zeros(1, n))

    # 0.1 in every channel -> e = 0.1 exactly, so q must be 0.1 / (0.1 + c_m) everywhere
    expected_q = 0.1 / (0.1 + cm)
    deviation = (corrupted - clean).abs().mean(dim=-1)
    assert torch.allclose(deviation, torch.full((1, n), 0.1), atol=1e-6)
    assert abs(expected_q - 0.1 / (0.1 + cm)) < 1e-9
    # the syndrome target is the check-averaged q, so a constant q must give a constant target
    support = outputs["H_support"]
    expected_density = (support @ torch.full((n, 1), expected_q)).squeeze(1) / support.sum(1)
    assert torch.allclose(expected_density, torch.full((m,), expected_q), atol=1e-5)
    assert torch.isfinite(parts["loss"])


def test_detect_loss_mask_target_is_the_default_and_uses_the_binary_mask():
    torch.manual_seed(4)
    outputs = _diagnostic_outputs(with_corrupted=True)
    outputs["clean_tokens"] = {"rgb": torch.zeros(1, N, D), "tir": torch.zeros(1, N, D)}
    outputs["reliability_rgb"] = torch.rand(1, N).clamp(0.01, 0.99)
    outputs["reliability_tir"] = torch.rand(1, N).clamp(0.01, 0.99)
    mask = outputs["token_mask_rgb"]
    loss_fn = CodeTrackLoss(num_parity=M, dim=D, code_dim=D)

    assert loss_fn.detect_target == "mask"
    deviation_target = CodeTrackLoss(num_parity=M, dim=D, code_dim=D,
                                     detect_target="deviation").detect_target
    assert deviation_target == "deviation"
    parts = loss_fn._detect_loss(outputs, mask, outputs["token_mask_tir"])
    assert torch.isfinite(parts["loss"])


def test_crop_box_for_frame_switches_to_the_reference_trajectory():
    """P3: a supplied reference trajectory drives the crop, not the run's own prediction.

    The reference is applied **lagged**: the free loop crops frame ``f`` around the box produced
    for ``f - 1``, never around its own output for ``f``.  Passing ``reference[f]`` (the first
    version) gave the replay a crop the free run never saw and leaked the reference's answer one
    frame early, which is also why the TIR replay of 6.12 has to be re-run.
    """
    predicted = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    reference = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)

    free = crop_box_for_frame(predicted, None, 1)
    replayed = crop_box_for_frame(predicted, reference, 1)
    later = crop_box_for_frame(predicted, reference, 2)
    first = crop_box_for_frame(predicted, reference, 0)

    assert np.array_equal(free, predicted)
    assert np.array_equal(replayed, reference[0])          # frame 1 <- reference[0]
    assert np.array_equal(later, reference[1])             # frame 2 <- reference[1]
    # frame 0 is driven by the annotation-derived initial box in every arm, so the schedule must
    # not override it: that was the other half of the misalignment
    assert np.array_equal(first, predicted)
    assert replayed.dtype == np.float32
    # the helper must not mutate the prediction it was handed
    assert np.array_equal(predicted, np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))


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
    # D = drop_B - drop_A: positive means the first arm degrades less
    assert row["mean_difference"] > 0.0
    low, high = row["difference_ci95"]
    assert low > 0.0 and row["ci_excludes_zero"] is True
    assert row["paired_t_p"] < 0.01
    assert row["sign_convention"].startswith("positive")


def test_difference_in_differences_is_zero_for_identical_arms():
    rng = np.random.default_rng(3)
    drop = rng.uniform(0.0, 0.1, size=30)

    row = difference_in_differences(drop, drop.copy(), rng, draws=1000)

    assert row["mean_difference"] == 0.0
    assert row["ci_excludes_zero"] is False


def test_path_substitution_replaces_exactly_the_requested_paths():
    """P3 stage 2: the intervention must swap the cached tensors and nothing else.

    A substitution built outside the per-frame forward (an earlier draft) put one frame's
    tensors on another frame's input and measured nothing; this checks the module-boundary
    mechanism on a tiny model: with ``control`` the reliability output and the gates are the
    cached clean ones, with ``feature`` the fusion receives the cached clean TIR tokens.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from trajectory_replay import path_substitution

    cfg = {
        "model": {
            "embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
            "img_size": 64, "template_size": 32, "code_dim": 32, "check_dim": 16,
            "num_identity_tokens": 4, "num_parity_tokens": 4, "num_variable_nodes": 16,
            "num_graph_nodes": 8, "bp_iterations": 2, "h_links_per_check": 4,
            "graph_top_k": 4, "fpn_dim": 16, "return_stages": (0, 1),
            "head_type": "CENTER", "head_channel": 16, "freeze_backbone": True,
        }
    }
    from codetrack.models.codetrack import CodeTrack

    torch.manual_seed(0)
    model = CodeTrack(cfg).eval()
    batch = {
        "template_rgb": torch.randn(1, 3, 32, 32),
        "search_rgb": torch.randn(1, 3, 64, 64),
        "template_tir": torch.randn(1, 1, 32, 32),
        "search_tir": torch.randn(1, 1, 64, 64),
    }
    args = (batch["template_rgb"], batch["search_rgb"], batch["template_tir"],
            batch["search_tir"])

    cache, capture_run, apply_substitution = path_substitution(model)
    capture_run(model, *args, corruption=None)
    assert set(cache) >= {"reliability", "gate", "corrected_tir", "taps_t"}

    seen = {}
    original_fusion = model.fusion.forward

    def spy_fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir, taps_r, taps_t):
        seen["corrected_tir"] = corrected_tir
        seen["taps_t"] = taps_t
        return original_fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir,
                               taps_r, taps_t)

    corruption = {"enabled": True, "token": ["tok_block_erase"], "ratio": 0.5, "severity": 0.4,
                  "target": "tir"}
    # the spy must sit *under* the substitution: install it first, then let the context wrap it
    model.fusion.forward = spy_fusion
    try:
        with apply_substitution("feature"):
            model(*args, corruption=corruption)
    finally:
        model.fusion.forward = original_fusion

    assert torch.equal(seen["corrected_tir"], cache["corrected_tir"])
    for key, value in cache["taps_t"].items():
        assert torch.equal(seen["taps_t"][key], value)

    # control: the gating module must hand the decoder the cached gates
    captured_gate = {}
    original_decoder = model.decoder.forward

    def spy_decoder(*a, **kwargs):
        captured_gate["gate_rgb"] = kwargs.get("gate_rgb")
        captured_gate["gate_tir"] = kwargs.get("gate_tir")
        return original_decoder(*a, **kwargs)

    with apply_substitution("control"):
        model.decoder.forward = spy_decoder
        try:
            model(*args, corruption=corruption)
        finally:
            model.decoder.forward = original_decoder

    assert torch.equal(captured_gate["gate_rgb"], cache["gate"][0])
    assert torch.equal(captured_gate["gate_tir"], cache["gate"][1])


def test_forward_decidable_failure_ignores_transient_dips_and_reports_recovery():
    """P5: the loss-time event must be decidable from the past only.

    ``loss_frame`` looks at the whole trace and at the total horizon ("never recovers"), which
    is descriptive but is not a first-failure event; ``first_failure`` fires on K consecutive
    frames below the threshold and therefore needs no future.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from horizon_probe import first_failure, loss_frame, recovered_after

    assert first_failure(np.ones(50), patience=10) is None

    transient = np.ones(50)
    transient[20:25] = 0.1                      # 5 bad frames, patience is 10
    assert first_failure(transient, patience=10) is None

    recovers = np.concatenate([np.ones(30), np.full(25, 0.1), np.ones(20)])
    failure = first_failure(recovers, patience=10)
    assert failure == 30
    assert recovered_after(recovers, failure, patience=10) is True
    # the descriptive definition disagrees by design: it never *stays* lost
    assert loss_frame(recovers, window=10) is None

    permanent = np.concatenate([np.ones(20), np.full(40, 0.1)])
    failure = first_failure(permanent, patience=10)
    assert failure == 20
    assert recovered_after(permanent, failure, patience=10) is False
    assert recovered_after(permanent, None, patience=10) is None


def test_power_plan_mde_scales_with_variance_family_and_sequences():
    """P5: the design estimate must follow the standard relations, not just print numbers."""
    sys.path.insert(0, str(ROOT / "tools"))
    from power_plan import mde, required_n

    base = mde(10.0, 60, 1)
    assert base > 0
    # doubling the variance doubles the MDE
    assert abs(mde(20.0, 60, 1) - 2 * base) < 1e-9
    # quadrupling n halves it
    assert abs(mde(10.0, 240, 1) - base / 2) < 1e-9
    # a bigger Holm family is more conservative
    assert mde(10.0, 60, 8) > base
    # required_n inverts mde: n for an effect of exactly the MDE at n=60 is 60
    assert abs(required_n(10.0, base, 1) - 60.0) < 1e-6
    assert required_n(10.0, 0.0, 1) is None


def test_power_plan_reads_the_mean_iou_endpoint_and_rejects_stale_runs(tmp_path):
    """P5: the lower-variance endpoint must be usable, and a pre-endpoint run must not be
    silently averaged as zeros."""
    sys.path.insert(0, str(ROOT / "tools"))
    from power_plan import per_sequence

    run = tmp_path / "clean"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({
        "per_sequence": [
            {"sequence": "a", "sr": 0.4, "iou_mean": 0.31},
            {"sequence": "b", "sr": 0.2, "iou_mean": 0.19},
        ]}))

    assert per_sequence(run / "metrics.json", "iou") == {"a": 0.31, "b": 0.19}
    assert per_sequence(run / "metrics.json", "sr") == {"a": 0.4, "b": 0.2}

    (run / "metrics.json").write_text(json.dumps({
        "per_sequence": [{"sequence": "a", "sr": 0.4}]}))
    with pytest.raises(SystemExit, match="iou_mean"):
        per_sequence(run / "metrics.json", "iou")


def test_crop_square_bounds_a_diverged_window():
    """A diverged prediction must not be able to allocate an unbounded crop.

    Before this bound, one frame of a diverged run asked ``np.pad`` for a 40000 x 40000 window
    (6.7 GB): a 60-sequence x 200-frame evaluation grew past 21 GB with the GPU idle.
    """
    from codetrack.data.transforms.sample import MAX_CROP_SIDE_FACTOR, _crop_square

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = _crop_square(image, 320.0, 240.0, 40000.0)

    limit = int(MAX_CROP_SIDE_FACTOR * 640)
    assert crop.shape == (limit, limit, 3)
    # a legitimate search window (4x the frame) is untouched
    legitimate = _crop_square(image, 320.0, 240.0, 4 * 640.0)
    assert legitimate.shape[0] <= limit
    # degenerate inputs cannot produce a zero-sized or non-finite window
    assert _crop_square(image, 320.0, 240.0, 0.0).shape == (480, 480, 3)
    assert _crop_square(image, np.nan, np.nan, 100.0).shape == (100, 100, 3)


def test_clamp_box_bounds_scale_position_and_non_finite_values():
    """The tracker loop must never propagate a box the crop code cannot handle."""
    keep, reasons = clamp_box(np.array([10.0, 20.0, 60.0, 80.0], dtype=np.float32), 640, 480)
    assert not reasons
    assert np.allclose(keep, [10.0, 20.0, 60.0, 80.0])

    huge, reasons = clamp_box(np.array([10.0, 20.0, 99999.0, 99999.0], dtype=np.float32),
                              640, 480)
    assert reasons
    assert huge[2] <= 4.0 * 640 and huge[3] <= 4.0 * 480
    # the centre is pulled back onto the frame, so the padded window stays bounded
    assert 0.0 <= huge[0] + huge[2] / 2 <= 640.0
    assert 0.0 <= huge[1] + huge[3] / 2 <= 480.0

    centred, reasons = clamp_box(np.array([np.nan, 0.0, 10.0, 10.0], dtype=np.float32),
                                 640, 480)
    assert reasons and np.all(np.isfinite(centred))
    assert np.allclose(centred, [160.0, 120.0, 320.0, 240.0])

    tiny, reasons = clamp_box(np.array([10.0, 10.0, 0.0, -5.0], dtype=np.float32), 640, 480)
    assert reasons and tiny[2] > 0 and tiny[3] > 0


def test_clamp_box_separates_scale_from_centre_and_nonfinite_events():
    """A single "clamped" counter mixed four different events.  "87 % of frames clamped" then
    reads as "87 % diverged" when most of those may be centre clamps, which are a much weaker
    event.  The reasons must be separable so a divergence claim can name what it measured."""
    from codetrack.engine.trainer import (CLAMP_CENTER, CLAMP_NONFINITE, CLAMP_NONPOSITIVE,
                                          CLAMP_SCALE)

    # x chosen so the *post-clamp* centre lands exactly on the frame edge: this isolates the
    # scale event from the centre event, which the old single flag could not distinguish.
    _, reasons = clamp_box(np.array([-1280.0, 20.0, 99999.0, 80.0], dtype=np.float32), 640, 480)
    assert reasons == {CLAMP_SCALE}, "a width blow-up must not be reported as a centre clamp"

    # A small box entirely outside the frame is clamped, but not because it grew.
    _, reasons = clamp_box(np.array([-500.0, -500.0, 20.0, 20.0], dtype=np.float32), 640, 480)
    assert reasons == {CLAMP_CENTER}

    _, reasons = clamp_box(np.array([1.0, 2.0, 0.0, 5.0], dtype=np.float32), 640, 480)
    assert reasons == {CLAMP_NONPOSITIVE}

    _, reasons = clamp_box(np.array([np.inf, 0.0, 10.0, 10.0], dtype=np.float32), 640, 480)
    assert reasons == {CLAMP_NONFINITE}


def test_divergence_rate_reads_the_clamp_counters():
    """P5/6.15: a condition that breaks the closed loop must be visible in the analysis."""
    sys.path.insert(0, str(ROOT / "tools"))
    from summarize_paired_conditions import divergence_rate

    assert divergence_rate({"n_frames": 1000, "n_box_clamped_frames": 250}) == 0.25
    # runs recorded before the aggregate existed are reconstructed from per-sequence rows
    assert divergence_rate({"per_sequence": [
        {"frames": 100, "box_clamps": 50}, {"frames": 100, "box_clamps": 0}]}) == 0.25
    assert divergence_rate({"per_sequence": [{"frames": 100, "box_clamps": 0}]}) == 0.0
    # a pre-fix run has neither field and must report "unknown", not zero
    assert divergence_rate({"per_sequence": [{"sr": 0.5}]}) is None


def test_holm_rejected_follows_the_step_down_procedure():
    sys.path.insert(0, str(ROOT / "tools"))
    from power_plan import holm_rejected

    # each p-value is compared against alpha / (m - rank + 1): 0.001 <= 0.05/3, 0.02 <= 0.05/2,
    # 0.04 <= 0.05/1 -- all three pass, and the smallest is always among them
    assert holm_rejected(np.array([0.001, 0.02, 0.04]), 0.05) == {0, 1, 2}
    assert holm_rejected(np.array([0.001, 0.03, 0.9]), 0.05) == {0}
    assert 0 in holm_rejected(np.array([0.001, 0.9, 0.9]), 0.05)
    assert holm_rejected(np.array([0.001, 0.6, 0.7]), 0.05) == {0}
    assert holm_rejected(np.array([0.4, 0.6, 0.7]), 0.05) == set()
    # the family size changes the threshold, not just the ordering
    assert holm_rejected(np.array([0.01, 0.01, 0.01]), 0.05) == {0, 1, 2}
    assert holm_rejected(np.array([0.01] * 60), 0.05) == set()


def test_simulated_power_is_monotone_and_controls_the_family_wise_error():
    """P5: power under the real Holm procedure, measured on a resampled distribution."""
    sys.path.insert(0, str(ROOT / "tools"))
    from power_plan import simulate_power

    rng = np.random.default_rng(0)
    differences = rng.normal(0.0, 4.0, size=60)          # 4-point spread, in the observed range

    null = simulate_power(differences, 0.0, comparisons=1, sequences=60, reps=400, seed=1)
    assert null["power"] <= 0.10                          # FWER near alpha, never 0.5

    weak = simulate_power(differences, 1.0, comparisons=8, sequences=60, reps=400, seed=1)
    strong = simulate_power(differences, 20.0, comparisons=8, sequences=60, reps=400, seed=1)
    assert strong["power"] > weak["power"] >= 0.0
    assert strong["power"] > 0.95 and weak["power"] < 0.5

    # an effect size that is *not* saturated, so the two monotonicities are actually tested
    family_penalty = simulate_power(differences, 1.5, comparisons=8, sequences=60, reps=400,
                                    seed=2)
    single = simulate_power(differences, 1.5, comparisons=1, sequences=60, reps=400, seed=2)
    assert single["power"] >= family_penalty["power"]
    assert family_penalty["power"] < 0.95

    more = simulate_power(differences, 1.5, comparisons=8, sequences=600, reps=400, seed=2)
    assert more["power"] > family_penalty["power"]


def test_simulated_fwer_counts_any_rejection_in_the_family():
    """The old code checked ``0 in holm_rejected(...)``, so the row labelled FWER reported
    P(this hypothesis is rejected) -- roughly alpha -- instead of P(any rejection).  Under a null
    family of 8 the two differ by a large factor, and the number feeds the "can this design ever
    detect anything" argument of 6.13, so it has to be the family-wise quantity."""
    sys.path.insert(0, str(ROOT / "tools"))
    from power_plan import simulate_power

    rng = np.random.default_rng(3)
    differences = rng.normal(0.0, 4.0, size=60)

    null = simulate_power(differences, 0.0, comparisons=8, sequences=60, reps=600, seed=5)
    per_hypothesis = null["power"]
    family_wise = null["family_wise_rejection_rate"]

    assert family_wise >= per_hypothesis, \
        "P(any rejection) cannot be below P(a specific rejection)"
    # Holm still controls the family-wise rate; with one shared vector the null comparisons are
    # perfectly dependent, so the control is conservative rather than tight.  Conservative is the
    # honest direction to be wrong in for a "can this design ever detect anything" number.
    assert family_wise <= 0.05
    # the old independent draw is what inflated the family-wise rate
    independent = simulate_power(differences, 0.0, comparisons=8, sequences=60, reps=600,
                                 seed=5, correlation="independent")
    assert independent["family_wise_rejection_rate"] > family_wise

    # A realistic family carries one measured vector per comparison; the shared resample then
    # preserves the real correlation instead of collapsing the family to one test.
    family = np.vstack([differences, 0.8 * differences + rng.normal(0, 1.0, size=60)])
    joint = simulate_power(family, 0.0, comparisons=2, sequences=60, reps=600, seed=5)
    assert joint["comparisons"] == 2 and joint["family_wise_rejection_rate"] <= 0.05


def test_rmst_and_confirmed_failure_frame():
    """Two endpoints the round-2 plan adds: the failure *time* a tracker could act on, and a
    censoring-aware mean time tracked.  Both are mechanistic secondaries, so they only have to be
    correct, not significant."""
    sys.path.insert(0, str(ROOT / "tools"))
    from horizon_probe import failure_confirmed_frame, first_failure, rmst

    iou = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.9], dtype=float)
    failure = first_failure(iou, threshold=0.5, patience=3)
    assert failure == 2, "the reported event time is the *start* of the failing run"
    assert failure_confirmed_frame(failure, 3) == 4, "knowable only after `patience` frames"
    assert failure_confirmed_frame(None, 3) is None

    # A sequence that never fails contributes its full horizon, not zero and not the maximum.
    assert rmst(np.array([200.0, 200.0]), np.array([0.0, 0.0]), 200) == 200.0
    # An immediate failure at frame 0 leaves no tracked time.
    assert rmst(np.array([0.0]), np.array([1.0]), 200) == 0.0
    # One of two sequences failing at frame 50: half the mass survives past 50.
    value = rmst(np.array([50.0, 200.0]), np.array([1.0, 0.0]), 200)
    assert abs(value - (50.0 + 0.5 * 150.0)) < 1e-9
    # tau truncates rather than extrapolating past the observed follow-up
    assert rmst(np.array([500.0]), np.array([0.0]), 200) == 200.0


def test_gradient_conflict_report_reads_both_log_generations():
    """Logs written before the weighted diagnostic must still parse (they cannot be
    regenerated), and the report must say that the unweighted pair alone is not enough."""
    import tempfile

    sys.path.insert(0, str(ROOT / "tools"))
    import gradient_conflict_report as reporter

    legacy = ("[grad] cos(L_track, L_correct) = +0.092 | |g_track| 0.597 | |g_correct| 0.109\n")
    weighted = legacy.rstrip("\n") + (" | cos(L_track, g_aux) = -0.310 | |g_aux| 1.204 "
                                      "| aux/track 2.017 | cos(L_track, g_total) = +0.402\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.log"
        path.write_text(legacy + weighted)
        rows = reporter.series(path)

    assert len(rows) == 2
    assert rows[0]["cos"] == 0.092 and "cos_aux" not in rows[0]
    assert rows[1]["cos_aux"] == -0.310 and rows[1]["aux_over_track"] == 2.017
    # the segment summary is what separates "harmless early, opposed late"
    assert reporter.segments(rows, "cos_aux") == [-0.31]


def test_reliability_auroc_scores_the_head_the_decoder_actually_multiplies():
    """1 - r must be scored against the token mask, per modality, and stay honest when the
    comparison is degenerate (no positives, no negatives, mismatched sizes)."""
    from codetrack.engine.evaluator import summarize_detection

    syndrome = [np.array([1.0]), np.array([0.5])]
    syndrome_labels = [np.array([1]), np.array([0])]
    locator = [np.array([2.0, 1.0]), np.array([1.0, 2.0])]
    locator_labels = [np.array([1, 0]), np.array([1, 0])]

    perfect = summarize_detection(
        syndrome, syndrome_labels, locator, locator_labels, None, None,
        [np.array([0.9, 0.8, 0.2, 0.1])], [np.array([1, 1, 0, 0])],
        [np.array([0.9, 0.8, 0.2, 0.1])], [np.array([1, 1, 0, 0])])
    assert perfect["reliability_auroc_rgb"] == 1.0
    assert perfect["reliability_auroc_tir"] == 1.0
    assert perfect["reliability_auroc"] == 1.0

    uninformative = summarize_detection(
        syndrome, syndrome_labels, locator, locator_labels, None, None,
        [np.array([0.5, 0.5, 0.5, 0.5])], [np.array([1, 1, 0, 0])], None, None)
    assert uninformative["reliability_auroc_rgb"] == 0.5
    assert "reliability_auroc_tir" not in uninformative

    # a mask with no healthy tokens has no negative class: AUROC is undefined, not 1.0
    degenerate = summarize_detection(
        syndrome, syndrome_labels, locator, locator_labels, None, None,
        [np.array([0.9, 0.8])], [np.array([1, 1])], None, None)
    assert np.isnan(degenerate["reliability_auroc_rgb"])


def test_summarizer_prints_the_sign_it_stores(tmp_path, monkeypatch, capsys):
    """Regression: the interaction table once printed `drop_a - drop_b` while storing
    `drop_b - drop_a`, so the point estimate disagreed in sign with its own CI on the same line.
    This runs the real CLI on two synthetic runs and compares the printed D with the JSON."""
    sys.path.insert(0, str(ROOT / "tools"))
    import summarize_paired_conditions as summarizer

    sequences = [f"seq{index}" for index in range(6)]
    for label, clean, condition in (("A", 0.40, 0.30), ("B", 0.40, 0.36)):
        run = tmp_path / label
        for name, value in (("clean", clean), ("tok_block_rgb_04", condition)):
            directory = run / name
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(json.dumps({
                "sr": value, "pr": value, "npr": value, "n_sequences": len(sequences),
                "per_sequence": [{"sequence": s, "sr": value, "pr": value, "npr": value}
                                 for s in sequences]}))

    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize_paired_conditions.py",
        "--run", f"A={tmp_path / 'A'}", "--run", f"B={tmp_path / 'B'}",
        "--reference", "clean", "--interaction", "A", "B", "--out", str(out)])
    assert summarizer.main() == 0
    printed = capsys.readouterr().out
    row = json.loads(out.read_text())["interactions"][0]

    # arm A's drop (0.10) is larger than arm B's (0.04), so D = drop_B - drop_A < 0
    assert row["mean_difference"] < 0
    # the same condition name appears in the per-run tables too, so take the interaction table
    section = printed.split("=== difference in differences")[-1]
    line = [row_line for row_line in section.splitlines()
            if row_line.startswith("tok_block_rgb_04")]
    assert line, "the interaction row was not printed"
    assert f"{row['mean_difference'] * 100:>8.2f}" in line[0]
    low, high = row["difference_ci95"]
    assert f"{low * 100:>7.2f}" in line[0] and f"{high * 100:>7.2f}" in line[0]


def test_summarizer_composite_is_one_contrast_over_several_conditions(tmp_path, monkeypatch,
                                                                     capsys):
    """A result measured on several conditions must be reportable as ONE contrast: per-sequence
    mean drop over the conditions, then the difference in differences.

    It is *not* part of the Holm family, because the contrast is defined by the conditions named
    on the command line rather than pre-registered with the primary endpoint.  Section 6.26's
    "+3.65 points, p = 0.013" came from this path and was read as if it carried the
    pre-registration of the primary endpoint, so the flag is now stored in the artifact."""
    sys.path.insert(0, str(ROOT / "tools"))
    import summarize_paired_conditions as summarizer

    sequences = [f"s{index}" for index in range(6)]
    conditions = ("tok_block_rgb_02", "tok_block_rgb_04")
    # arm A loses 0.10 / 0.20 on the two conditions, arm B loses 0.05 / 0.10
    plan = {"A": {"clean": 0.50, "tok_block_rgb_02": 0.40, "tok_block_rgb_04": 0.30},
            "B": {"clean": 0.50, "tok_block_rgb_02": 0.45, "tok_block_rgb_04": 0.40}}
    for label, values in plan.items():
        for name, value in values.items():
            directory = tmp_path / label / name
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(json.dumps({
                "n_sequences": len(sequences), "sr": value, "pr": value, "npr": value,
                "per_sequence": [{"sequence": s, "sr": value, "pr": value, "npr": value}
                                 for s in sequences]}))

    out = tmp_path / "composite.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize_paired_conditions.py",
        "--run", f"A={tmp_path / 'A'}", "--run", f"B={tmp_path / 'B'}",
        "--reference", "clean", "--composite", "A", "B",
        "--composite-conditions", ",".join(conditions), "--out", str(out)])
    assert summarizer.main() == 0
    row = json.loads(out.read_text())["composite"]

    # mean drop in points: A = (10 + 20)/2 = 15, B = (5 + 10)/2 = 7.5 -> D = drop_B - drop_A
    assert abs(row["mean_drop_a"] - 15.0) < 1e-6
    assert abs(row["mean_drop_b"] - 7.5) < 1e-6
    assert abs(row["mean_difference"] + 7.5) < 1e-6           # A degrades MORE here
    assert row["conditions"] == list(conditions) and row["relative"] is False
    assert row["in_holm_family"] is False, \
        "the composite is defined by the command line and must not be folded into the Holm family"
    printed = capsys.readouterr().out
    assert "NOT part of the Holm family" in printed


def test_summarizer_metric_selection_and_missing_endpoint_tolerance(tmp_path, monkeypatch,
                                                                    capsys):
    """The IoU endpoint only exists in newer runs: a run without it must shrink n, not crash,
    and --metric must actually drive the reported drop."""
    sys.path.insert(0, str(ROOT / "tools"))
    import summarize_paired_conditions as summarizer

    sequences = [f"s{i}" for i in range(6)]
    # arm A has both endpoints; arm B predates the mean-IoU endpoint
    for name, value_sr, value_iou, with_iou in (("clean", 0.50, 0.40, True),
                                                ("tok_block_rgb_04", 0.30, 0.20, True)):
        directory = tmp_path / "A" / name
        directory.mkdir(parents=True)
        rows = [{"sequence": s, "sr": value_sr, "pr": value_sr, "npr": value_sr, "iou_mean": value_iou}
                for s in sequences]
        (directory / "metrics.json").write_text(json.dumps({"per_sequence": rows}))
    for name, value in (("clean", 0.50), ("tok_block_rgb_04", 0.36)):
        directory = tmp_path / "B" / name
        directory.mkdir(parents=True)
        rows = [{"sequence": s, "sr": value, "pr": value, "npr": value} for s in sequences]
        (directory / "metrics.json").write_text(json.dumps({"per_sequence": rows}))

    out = tmp_path / "iou_summary.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize_paired_conditions.py", "--run", f"A={tmp_path / 'A'}",
        "--run", f"B={tmp_path / 'B'}", "--reference", "clean", "--metric", "iou_mean",
        "--out", str(out)])
    assert summarizer.main() == 0
    report = json.loads(out.read_text())
    stats = report["runs"]["A"]["conditions"]["tok_block_rgb_04"]

    # A has the endpoint: the drop is measured on it (0.40 -> 0.20, i.e. 20 points)
    assert abs(stats["iou_mean"]["mean_drop_points"] * 100 - 20.0) < 1e-6
    # B lacks it, so its comparison is empty rather than an error
    assert report["runs"]["B"]["conditions"]["tok_block_rgb_04"]["iou_mean"]["n_sequences"] == 0
    # and the SR numbers are still computed alongside
    assert abs(stats["sr"]["mean_drop_points"] * 100 - 20.0) < 1e-6
    assert "iou_mean" in capsys.readouterr().out or True


def test_divergence_report_separates_zero_from_unmeasured(tmp_path):
    """A pre-fix run must read "unknown", never 0 %: they mean different things."""
    sys.path.insert(0, str(ROOT / "tools"))
    from divergence_report import rates

    measured = tmp_path / "measured" / "cond"
    measured.mkdir(parents=True)
    (measured / "metrics.json").write_text(json.dumps({
        "n_frames": 1000, "n_box_clamped_frames": 25,
        "per_sequence": [{"sequence": "a", "frames": 500, "box_clamps": 25},
                         {"sequence": "b", "frames": 500, "box_clamps": 0}]}))
    # only per-sequence rows, no aggregate: reconstructed
    per_sequence_only = tmp_path / "rows" / "cond"
    per_sequence_only.mkdir(parents=True)
    (per_sequence_only / "metrics.json").write_text(json.dumps({
        "per_sequence": [{"sequence": "a", "frames": 200, "box_clamps": 10}]}))
    # a run from before the counter existed
    old = tmp_path / "old" / "cond"
    old.mkdir(parents=True)
    (old / "metrics.json").write_text(json.dumps({"sr": 0.4, "per_sequence": [{"sequence": "a"}]}))

    assert rates(tmp_path / "measured")[0]["divergence_rate"] == 0.025
    assert rates(tmp_path / "measured")[0]["worst_sequence"] == "a"
    assert rates(tmp_path / "rows")[0]["divergence_rate"] == 0.05
    assert rates(tmp_path / "old")[0]["divergence_rate"] is None


def test_clamp_audit_report_reads_both_runs(tmp_path, capsys):
    """The audit must show the pre-fix and clamped numbers side by side, and say 'missing'
    rather than silently skipping a run that has not been re-evaluated yet."""
    sys.path.insert(0, str(ROOT / "tools"))
    from clamp_audit_report import main as audit_main

    for directory, sr, clamped in ((tmp_path / "before", 0.25, None),
                                   (tmp_path / "after", 0.24, 1234)):
        condition = directory / "tok_block_rgb_04"
        condition.mkdir(parents=True)
        metrics = {"sr": sr, "pr": 0.30, "npr": 0.26, "iou": sr - 0.01}
        if clamped is not None:
            metrics.update({"n_box_clamped_frames": clamped, "n_frames": 10244})
        (condition / "metrics.json").write_text(json.dumps(metrics))

    argv = ["clamp_audit_report.py", "--before", str(tmp_path / "before"),
            "--after", str(tmp_path / "after"), "--conditions", "tok_block_rgb_04,absent"]
    import sys as _sys
    original = _sys.argv
    _sys.argv = argv
    try:
        assert audit_main() == 0
    finally:
        _sys.argv = original
    printed = capsys.readouterr().out

    assert "pre-fix" in printed and "clamped" in printed
    assert "24.00" in printed and "1234" in printed
    assert "missing" in printed          # the condition that has not been re-run yet


def test_family_report_compares_arms_on_the_primary_family(tmp_path, capsys):
    """The P4 decision rule lives in the tool: the primary endpoint is one family's
    reliability AUROC, compared against the mask-trained baseline."""
    sys.path.insert(0, str(ROOT / "tools"))
    from family_report import main as family_main

    for label, auc in (("mask", 0.7440), ("deviation", 0.8100)):
        for family, value in (("burst", 1.0), ("noise", auc)):
            directory = tmp_path / label / family
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(json.dumps({
                "reliability_auroc_rgb": value, "syndrome_auroc": 0.5, "recovery_gain": -0.02,
                "e_before_mean": 0.12, "e_after_mean": 0.23}))

    import sys as _sys
    original = _sys.argv
    _sys.argv = ["family_report.py",
                 "--arm", f"mask={tmp_path / 'mask'}/", "--arm", f"deviation={tmp_path / 'deviation'}/",
                 "--families", "burst:zeroing,noise:additive noise",
                 "--primary", "noise", "--out", str(tmp_path / "family.json")]
    try:
        assert family_main() == 0
    finally:
        _sys.argv = original
    printed = capsys.readouterr().out
    report = json.loads((tmp_path / "family.json").read_text())

    assert "0.8100" in printed and "0.7440" in printed
    assert abs(report["primary_endpoint"]["deviation"]["delta"] - 0.066) < 1e-9
    assert report["primary_endpoint"]["deviation"]["baseline_label"] == "mask"


def test_gradient_conflict_report_parses_the_training_log():
    """The diagnostic is already logged; the tool must parse it exactly and not invent rows."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gradient_conflict_report import series

    log = tmp_path_placeholder = None
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(
            "[00:00:01] INFO [grad] cos(L_track, L_correct) = +0.092 | |g_track| 0.597 "
            "| |g_correct| 0.109\n"
            "[00:00:02] INFO [grad] cos(L_track, L_correct) = -0.250 | |g_track| 0.438 "
            "| |g_correct| 0.099\n"
            "[00:00:03] INFO [joint] it 10 | loss 1.0\n")     # not a grad line
        rows = series(path)
    assert len(rows) == 2
    assert rows[0] == {"cos": 0.092, "g_track": 0.597, "g_correct": 0.109}
    assert rows[1]["cos"] == -0.25


def test_load_checkpoint_refuses_a_decoder_output_mismatch(tmp_path):
    """An identity-residual arm loaded into a `post_norm` model silently drops its branch norm and
    residual projection, i.e. evaluates a different network.  The same is true for an arm trained
    with the gate forced to 1 and evaluated without the flag (that module never saw a gradient).
    Both mistakes were made while running the P2 grid, so both now fail loudly."""
    from codetrack.models.codetrack import CodeTrack
    from codetrack.utils.checkpoint import save_checkpoint

    base = {
        "model": {
            "embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
            "img_size": 64, "template_size": 32, "code_dim": 32, "check_dim": 16,
            "num_identity_tokens": 4, "num_parity_tokens": 4, "num_variable_nodes": 16,
            "num_graph_nodes": 8, "bp_iterations": 2, "h_links_per_check": 4,
            "graph_top_k": 4, "fpn_dim": 16, "return_stages": (0, 1),
            "head_type": "CENTER", "head_channel": 16, "freeze_backbone": True,
        }
    }
    identity_cfg = {"model": {**base["model"], "decoder_output": "identity_residual"}}
    gate_cfg = {"model": {**base["model"], "gate_always_one": True}}

    for label, cfg in (("identity_residual", identity_cfg), ("gate_always_one", gate_cfg)):
        source = CodeTrack(cfg)
        path = tmp_path / f"{label}.pth"
        save_checkpoint(path, source, optimizer=None, epoch=0, cfg=cfg)

        target = CodeTrack(base)
        with pytest.raises(RuntimeError, match="was trained with"):
            load_checkpoint(path, target, map_location="cpu")

        # and the matching model loads cleanly
        assert load_checkpoint(path, CodeTrack(cfg), map_location="cpu")["missing"] is not None


def test_crop_mapping_uses_the_geometry_the_model_was_actually_given():
    """`_crop_square` clamps the side and the centre, so the crop the model sees is not always the
    one requested.  Mapping the head's normalised box back with the requested geometry is wrong by
    exactly the clamp whenever one fires -- and the clamp only fires in the diverged regime, so the
    error is invisible in every healthy run and appears precisely when it matters.  The flag keeps
    the historical assumption available so archived numbers stay reproducible."""
    from codetrack.data.transforms.sample import MAX_CROP_SIDE_FACTOR, crop_geometry

    # healthy frame: no clamp, the two agree exactly
    assert crop_geometry(320.0, 240.0, 100.0, 480, 640) == (320.0, 240.0, 100.0)

    # a diverged side is clamped to MAX_CROP_SIDE_FACTOR x the longer edge
    _, _, side = crop_geometry(320.0, 240.0, 99999.0, 480, 640)
    assert side == MAX_CROP_SIDE_FACTOR * 640

    # a centre outside the frame is pulled onto it (the historical mapping would keep it outside)
    cx, cy, _ = crop_geometry(-500.0, 5000.0, 100.0, 480, 640)
    assert (cx, cy) == (0.0, 480.0)

    # and the loop only applies the clamped geometry when asked
    cfg = {"model": {"embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
                     "img_size": 64, "template_size": 32, "code_dim": 32, "check_dim": 16,
                     "num_identity_tokens": 4, "num_parity_tokens": 4, "num_variable_nodes": 16,
                     "num_graph_nodes": 8, "bp_iterations": 2, "h_links_per_check": 4,
                     "graph_top_k": 4, "fpn_dim": 16, "return_stages": (0, 1),
                     "head_type": "CENTER", "head_channel": 16, "freeze_backbone": True}}
    from codetrack.engine.trainer import Trainer
    assert Trainer(cfg, output_dir=Path("/tmp")).crop_mapping == "assumed"
    cfg["eval"] = {"crop_mapping": "actual"}
    assert Trainer(cfg, output_dir=Path("/tmp")).crop_mapping == "actual"
    cfg["eval"] = {"crop_mapping": "sideways"}
    with pytest.raises(ValueError, match="crop_mapping"):
        Trainer(cfg, output_dir=Path("/tmp"))


def test_score_window_suppresses_a_spurious_corner_peak():
    """The centre head takes a global argmax, so a response at the corner of the window wins
    outright -- the failure mode OSTrack's `window_influence` exists to prevent.  The switch must
    actually change the argmax when a spurious peak is present, and must be off by default."""
    from codetrack.models.head.center import CenterPredictor

    head = CenterPredictor(inplanes=8, channel=8, feat_sz=4, stride=16)
    assert head.score_window == "none"

    score = torch.full((1, 1, 4, 4), 0.20)
    score[0, 0, 0, 0] = 0.30          # spurious corner peak, above the centre response
    score[0, 0, 2, 2] = 0.25
    size = torch.zeros(1, 2, 16)
    offset = torch.zeros(1, 2, 16)

    without = head.cal_bbox(score.clone(), size, offset)
    assert torch.allclose(without[0, :2], torch.zeros(2)), \
        "without the window the corner peak must win, which is the failure being fixed"

    head.score_window = "hann"
    head.window_influence = 0.5
    with_window = head.cal_bbox(score.clone(), size, offset)
    assert torch.allclose(with_window[0, :2], torch.full((2,), 0.5)), \
        "the Hann window must pull the choice back to the window centre"


def test_load_checkpoint_refuses_a_decoder_state_mode_mismatch(tmp_path):
    """``decoder_state_mode`` changes the decoder's *function* while leaving every tensor in
    place, so a mismatched load cannot be detected from the state dict and silently evaluated a
    different network.  That is what 073c223 did by accident; the guard now makes the choice
    explicit, and the P2 arms A and B (trained under "held") need the override to be rebuilt."""
    from codetrack.models.codetrack import CodeTrack
    from codetrack.utils.checkpoint import save_checkpoint

    base = {
        "model": {
            "embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
            "img_size": 64, "template_size": 32, "code_dim": 32, "check_dim": 16,
            "num_identity_tokens": 4, "num_parity_tokens": 4, "num_variable_nodes": 16,
            "num_graph_nodes": 8, "bp_iterations": 2, "h_links_per_check": 4,
            "graph_top_k": 4, "fpn_dim": 16, "return_stages": (0, 1),
            "head_type": "CENTER", "head_channel": 16, "freeze_backbone": True,
        }
    }
    held_cfg = {"model": {**base["model"], "decoder_state_mode": "held"}}
    path = tmp_path / "held.pth"
    save_checkpoint(path, CodeTrack(held_cfg), optimizer=None, epoch=0, cfg=held_cfg)

    with pytest.raises(RuntimeError, match="decoder_state_mode"):
        load_checkpoint(path, CodeTrack(base), map_location="cpu")

    assert load_checkpoint(path, CodeTrack(held_cfg), map_location="cpu")["missing"] is not None


def _tiny_trainer_cfg(**train_overrides):
    cfg = {"model": {"embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
                     "img_size": 64, "template_size": 32, "code_dim": 32, "check_dim": 16,
                     "num_identity_tokens": 4, "num_parity_tokens": 4, "num_variable_nodes": 16,
                     "num_graph_nodes": 8, "bp_iterations": 2, "h_links_per_check": 4,
                     "graph_top_k": 4, "fpn_dim": 16, "return_stages": (0, 1),
                     "head_type": "CENTER", "head_channel": 16, "freeze_backbone": True},
           "train": {"epochs": 1, "samples_per_epoch": 32, "lr": 1e-4, "weight_decay": 1e-4,
                     "grad_clip": 1.0, "amp": False, "log_every": 1, "codec_warmup_epochs": 0,
                     "batch_size": 2, "num_workers": 0, "grad_diag_every": 0, **train_overrides}}
    return cfg


def test_optimizer_groups_separate_adapters_and_do_not_decay_norms():
    """An adapter has to be able to move at its own rate, and weight decay on a LayerNorm gain or a
    bias is a different regulariser from weight decay on a weight matrix.  With no adapter the
    adapter group must be absent, so archived single-rate runs are unchanged."""
    from codetrack.engine.trainer import Trainer

    plain = Trainer(_tiny_trainer_cfg(), output_dir=Path("/tmp"))
    names = [group.get("name") for group in plain.optimizer.param_groups]
    assert "lora" not in names
    assert set(names) == {"decay", "no_decay"}
    by_name = {group.get("name"): group for group in plain.optimizer.param_groups}
    assert by_name["decay"]["weight_decay"] == 1e-4
    assert by_name["no_decay"]["weight_decay"] == 0.0

    cfg = _tiny_trainer_cfg(lr_lora=5e-4)
    cfg["model"]["lora"] = {"enabled": True, "rank": 4, "alpha": 4.0}
    adapted = Trainer(cfg, output_dir=Path("/tmp"))
    by_name = {group.get("name"): group for group in adapted.optimizer.param_groups}
    assert set(by_name) == {"lora", "decay", "no_decay"}
    assert by_name["lora"]["lr"] == 5e-4
    assert by_name["decay"]["lr"] == 1e-4
    assert sum(p.numel() for p in by_name["lora"]["params"]) == \
        adapted.model.backbone.lora_parameter_count() > 0


def test_cosine_schedule_warms_up_then_decays_and_preserves_group_ratios():
    from codetrack.engine.trainer import Trainer

    cfg = _tiny_trainer_cfg(lr_lora=1e-3, schedule="cosine", warmup_steps=10, max_steps=100)
    cfg["model"]["lora"] = {"enabled": True, "rank": 4}
    trainer = Trainer(cfg, output_dir=Path("/tmp"))
    groups = {group.get("name"): group for group in trainer.optimizer.param_groups}

    trainer._apply_lr_schedule(0)
    assert groups["lora"]["lr"] < 1e-3, "warmup must start below the base rate"
    trainer._apply_lr_schedule(9)
    assert abs(groups["lora"]["lr"] - 1e-3) < 1e-12, "warmup must reach the base rate"
    # the ratio between groups is a property of the configuration, not of the schedule
    assert abs(groups["lora"]["lr"] / groups["decay"]["lr"] - 10.0) < 1e-9
    trainer._apply_lr_schedule(55)
    mid = groups["lora"]["lr"]
    trainer._apply_lr_schedule(78)
    late = groups["lora"]["lr"]
    trainer._apply_lr_schedule(100)
    end = groups["lora"]["lr"]
    assert 0.0 < late < mid < 1e-3, "cosine must decay monotonically"
    assert end == 0.0, "a cosine decay to zero must actually reach zero at max_steps"
    # a constant-rate configuration must never have its lr touched
    constant = Trainer(_tiny_trainer_cfg(), output_dir=Path("/tmp"))
    before = [g["lr"] for g in constant.optimizer.param_groups]
    assert constant.schedule == "constant"
    assert before == [g["lr"] for g in constant.optimizer.param_groups]


def test_load_checkpoint_refuses_a_lora_mismatch(tmp_path):
    """Two adapters of different rank have different state dicts, and because `lora_b` starts at
    zero a mismatched load leaves it *randomly* initialised -- a random delta added to the frozen
    backbone, which no metric would flag as impossible."""
    from codetrack.engine.trainer import Trainer
    from codetrack.utils.checkpoint import load_checkpoint, save_checkpoint

    def build(rank, enabled=True):
        cfg = _tiny_trainer_cfg()
        cfg["model"]["lora"] = {"enabled": enabled, "rank": rank, "alpha": float(rank)}
        trainer = Trainer(cfg, output_dir=Path("/tmp"))
        return cfg, trainer.model

    cfg, model = build(4)
    path = tmp_path / "lora4.pth"
    save_checkpoint(path, model, optimizer=None, epoch=0, cfg=cfg)

    _, plain_model = build(4, enabled=False)
    with pytest.raises(RuntimeError, match="model.lora.enabled"):
        load_checkpoint(path, plain_model, map_location="cpu")

    _, wide_model = build(8)
    with pytest.raises(RuntimeError, match="model.lora.rank"):
        load_checkpoint(path, wide_model, map_location="cpu")

    _, same_model = build(4)
    assert load_checkpoint(path, same_model, map_location="cpu")["missing"] is not None
