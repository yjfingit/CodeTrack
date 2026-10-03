"""Architecture-property tests.

Each test locks one invariant that the review identified as violated in ``b530d9a``.
They run on a deliberately tiny backbone (2 blocks, dim 64) so the whole file finishes in
seconds on CPU and needs no pretrained checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codetrack.models.codetrack import CodeTrack  # noqa: E402


def tiny_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"model": {
        "embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 16,
        "img_size": 64, "template_size": 32,          # 4x4 = 16 search / 2x2 = 4 template
        "code_dim": 32, "check_dim": 16,
        "num_identity_tokens": 4, "num_parity_tokens": 4,
        "num_variable_nodes": 16, "num_graph_nodes": 8,
        "bp_iterations": 2, "h_links_per_check": 4, "graph_top_k": 4,
        "fpn_dim": 16, "return_stages": (0, 1),
        "head_type": "CENTER", "head_channel": 16,
        "freeze_backbone": True,
    }}
    cfg["model"].update(overrides)
    return cfg


def dummy_batch(model: CodeTrack, b: int = 2):
    m = model.cfg["model"]
    s, t = m["img_size"], m["template_size"]
    dim = m["embed_dim"]
    return {
        "template_rgb": torch.randn(b, 3, t, t),
        "search_rgb": torch.randn(b, 3, s, s),
        "template_tir": torch.randn(b, 1, t, t),
        "search_tir": torch.randn(b, 1, s, s),
    }


# --------------------------------------------------------------------- identity
def test_identity_tokens_are_permutation_invariant():
    """P0-6: the identity codewords must come from the *appended* learnable queries.

    The 16/4 queries are the last positions of the 128+4 query sequence, so permuting the
    template tokens must not change them.  Reading ``q[:, :K]`` instead would return
    template tokens and break this.
    """
    torch.manual_seed(0)
    model = CodeTrack(tiny_cfg())
    cb = model.codebook
    z_r = torch.randn(1, 4, 64)
    z_t = torch.randn(1, 4, 64)

    id_a, _ = cb.encode_identity(z_r, z_t)
    id_b, _ = cb.encode_identity(z_r[:, torch.randperm(4)], z_t)

    assert torch.allclose(id_a, id_b, atol=1e-4), \
        "identity codewords changed when only the template token order changed"


def test_identity_queries_receive_gradient():
    model = CodeTrack(tiny_cfg())
    z = torch.randn(1, 4, 64, requires_grad=True)
    identity, _ = model.codebook.encode_identity(z, z)
    identity.sum().backward()
    assert model.codebook.identity_queries.grad.abs().sum() > 0
    assert z.grad.abs().sum() > 0


# -------------------------------------------------------------------- incidence
def test_single_shared_incidence_is_row_normalised():
    """P0-1: exactly one H, non-negative and row-normalised, shared by every stage."""
    model = CodeTrack(tiny_cfg())
    h = model.decoder.matrix()
    m, n = model.num_parity, model.num_variables
    assert h.shape == (m, n)
    assert (h >= 0).all(), "parity-check weights must stay non-negative (softplus)"
    assert torch.allclose(h.sum(dim=1), torch.ones(m), atol=1e-5)


def test_parity_check_has_min_column_degree():
    """P1-4: no variable may be watched by fewer than `min_column_degree` checks."""
    model = CodeTrack(tiny_cfg())
    support = model.decoder.connectivity()
    col = support.sum(dim=0)
    assert (col >= 2).all(), f"variables with degree < 2: {(col < 2).sum().item()}"


def test_exposed_H_is_the_one_used_by_the_decoder():
    """The H handed to the loss must be the same object the decoder propagates over."""
    model = CodeTrack(tiny_cfg())
    out = model(**dummy_batch(model), corruption={"enabled": False})
    h_out = out["H"]
    assert torch.allclose(h_out, model.decoder.matrix(), atol=1e-6)
    assert h_out.shape == (model.num_parity, model.num_variables)


# --------------------------------------------------------------------- locator
def test_locator_votes_along_the_shared_incidence():
    """P0-3: the scattered vote must be exactly ``H^T`` applied to the syndrome.

    The syndrome is frame-centred before the vote (see :class:`ErrorLocator`): ``L_detect``
    pins its absolute level to the corruption density, and reusing the raw tensor for a
    variable-level BCE put the two terms in a tug-of-war over one variable.  The property
    that must survive is therefore "the vote is an ``H^T``-weighted combination of the
    syndrome", not "it is ``s @ H`` bit for bit".
    """
    model = CodeTrack(tiny_cfg())
    b, m, n, k = 2, model.num_parity, model.num_variables, model.num_graph_nodes
    syndrome = torch.rand(b, 1, m)
    identity_map = torch.rand(b, k)
    node_index = torch.stack([torch.randperm(n)[:k] for _ in range(b)])

    h = model.decoder.matrix()
    out = model.locator(syndrome, identity_map, node_index, n, H=h)

    s = syndrome.flatten(1)
    centred = s - s.mean(dim=1, keepdim=True)
    scale = centred.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    expected = ((0.5 + 0.5 * centred / scale) @ h).clamp(0.0, 1.0)
    assert torch.allclose(out["locator_scattered"], expected, atol=1e-6)

    # and the incidence must still be what decides the weights: a different support gives a
    # different vote (using the all-ones matrix, which no random draw equals)
    out_all = model.locator(syndrome, identity_map, node_index, n,
                            H=torch.full_like(h, 1.0 / n))
    assert not torch.allclose(out_all["locator_scattered"],
                              out["locator_scattered"], atol=1e-6)


def test_locator_vote_is_invariant_to_the_syndrome_absolute_level():
    """Two frames that differ only by a constant offset must produce the same vote.

    This is the decoupling that stops the localization BCE from fighting the density
    calibration: adding 0.3 to every check leaves "which checks are worse than average"
    unchanged, so the localization loss has no opinion about the absolute level.
    """
    model = CodeTrack(tiny_cfg())
    b, m, n, k = 2, model.num_parity, model.num_variables, model.num_graph_nodes
    s = torch.rand(b, m) * 0.4                     # keep room to shift upward, no clamp
    identity_map = torch.rand(b, k)
    node_index = torch.stack([torch.randperm(n)[:k] for _ in range(b)])
    h = model.decoder.matrix()

    a = model.locator(s.view(b, 1, m), identity_map, node_index, n, H=h)
    shifted = model.locator((s + 0.3).view(b, 1, m), identity_map, node_index, n, H=h)
    assert torch.allclose(a["locator_scattered"], shifted["locator_scattered"], atol=1e-5)


# ----------------------------------------------------------------- bounds
def test_gates_severity_and_locator_stay_in_unit_interval():
    """P1-6: identity_map is a probability, so severity / locator / gates are bounded."""
    model = CodeTrack(tiny_cfg()).eval()
    out = model(**dummy_batch(model), corruption={"enabled": False})
    for key in ("severity", "gate_rgb", "gate_tir", "locator_scattered",
                "fpn_conf", "syndrome"):
        v = out[key]
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, f"{key} out of [0, 1]"


def test_identity_map_is_a_probability():
    model = CodeTrack(tiny_cfg()).eval()
    out = model(**dummy_batch(model), corruption={"enabled": False})
    assert float(out["identity_map"].min()) >= 0.0
    assert float(out["identity_map"].max()) <= 1.0


# --------------------------------------------------------- per-modality masks
def test_corruption_masks_are_independent():
    """P0-7: damaging RGB must not label TIR tokens as corrupted."""
    model = CodeTrack(tiny_cfg())
    x_r = torch.randn(2, model.num_variables, 64)
    x_t = torch.randn(2, model.num_variables, 64)
    cfg = {"enabled": True, "target": "rgb", "ratio": 0.5, "token": ["tok_random_erase"]}

    _, _, m_r, m_t = model._maybe_corrupt(x_r, x_t, cfg)
    assert float(m_r.sum()) > 0
    assert float(m_t.sum()) == 0.0

    cfg["target"] = "tir"
    _, _, m_r2, m_t2 = model._maybe_corrupt(x_r, x_t, cfg)
    assert float(m_r2.sum()) == 0.0
    assert float(m_t2.sum()) > 0


# ------------------------------------------------------------------ baselines
def test_mlp_baseline_matches_bp_shapes_and_stays_parameter_matched():
    """P1-3: the ablation decoder must be usable and roughly parameter-matched."""
    bp = CodeTrack(tiny_cfg())
    mlp_cfg = tiny_cfg()
    mlp_cfg["model"]["decoder_mode"] = "mlp"
    mlp = CodeTrack(mlp_cfg)

    assert mlp.decoder.mlp_blocks is not None and len(mlp.decoder.mlp_blocks) == 2
    bp_params = sum(p.numel() for p in bp.decoder.parameters())
    mlp_params = sum(p.numel() for p in mlp.decoder.parameters())
    # the width is derived from dim, so the ratio holds at every scale; the bounds are
    # loose on the 64-dim toy config and tight (~1.0) at the real 768-dim one
    assert 0.4 < mlp_params / bp_params < 2.5, \
        f"baseline not parameter-matched: mlp={mlp_params} bp={bp_params}"

    out = mlp.eval()(**dummy_batch(mlp))
    assert out["corrected_rgb"].shape == (2, mlp.num_variables, 64)


def test_fpn_path_can_be_disabled():
    """P0-4: the clean-FPN shortcut must be removable for the causal ablation."""
    model = CodeTrack(tiny_cfg(use_fpn=False))
    out = model.eval()(**dummy_batch(model))
    assert out["fpn_conf"] is None


def test_fpn_gate_starts_nearly_closed():
    """P0-4: the gate is initialised so the FPN shortcut starts almost shut."""
    model = CodeTrack(tiny_cfg())
    conf = torch.sigmoid(model.fusion.fpn_gate(torch.zeros(1, 64, 4, 4)))
    assert float(conf.mean()) < 0.2


# ------------------------------------------------------------------ backbone
def test_backbone_is_fully_frozen():
    """The figure says frozen: no trainable parameter may remain in the backbone."""
    model = CodeTrack(tiny_cfg())
    for p in model.backbone.parameters():
        assert not p.requires_grad, "backbone has a trainable parameter"


def test_ostrack_state_dict_keys_load_without_renaming():
    """The ported ViT must expose the same tensor names OSTrack ships."""
    from codetrack.models.backbone import build_backbone

    backbone = build_backbone({"backbone": {"embed_dim": 64, "depth": 2, "num_heads": 4,
                                           "patch_size": 16, "img_size": 64,
                                           "template_size": 32, "pretrain_img_size": 64,
                                           "freeze": True}})
    keys = set(backbone.state_dict())
    for expected in ("patch_embed_rgb.proj.weight", "pos_embed", "pos_embed_z",
                     "pos_embed_x", "blocks.0.attn.qkv.weight", "norm.weight"):
        assert expected in keys, f"missing OSTrack-compatible key {expected}"


# ------------------------------------------------- review round 2: syndrome target
def test_syndrome_target_is_density_not_binary_hit():
    """P0-A: the syndrome must be supervised with a *graded* target.

    With ``min_column_degree >= 2`` the number of edges is bounded below by
    ``2*N/M``, so a check watches many variables and ``1[(H@M)>0]`` is ~always 1 under
    realistic corruption.  The training target must therefore be the corruption
    **density** inside each check's neighbourhood, which stays informative.
    """
    from codetrack.engine.losses import CodeTrackLoss

    torch.manual_seed(0)
    m, n = 4, 16
    support = torch.zeros(m, n)
    for j in range(m):
        support[j, torch.randperm(n)[:6]] = 1.0
    h = support / support.sum(dim=1, keepdim=True)

    mask = torch.zeros(2, n)
    mask[0, :3] = 1.0            # 3 of 16 = 18.75%
    mask[1, :1] = 1.0

    out = {"reliability_rgb": torch.full((2, n), 0.9),
           "reliability_tir": torch.full((2, n), 0.9),
           "syndrome": torch.full((2, 1, m), 0.2),
           "locator_scattered": torch.full((2, n), 0.01),
           "H": h, "H_support": support}

    loss = CodeTrackLoss(lambda_correct=0.0)
    det = loss._detect_loss(out, mask, torch.zeros_like(mask))

    degree = support.sum(dim=1)
    density = (support @ mask.t()).t() / degree
    assert density.max() < 0.5, "test setup: binary-hit label would already be degenerate"

    # a *constant* syndrome must be punished more when it is farther from the density
    off = float((density - 0.2).abs().mean())
    near = out["syndrome"].clone()
    near.fill_(float(density.mean()))
    det_near = loss._detect_loss({**out, "syndrome": near}, mask, torch.zeros_like(mask))
    assert det["syndrome"] > det_near["syndrome"]


def test_syndrome_loss_rewards_matching_density():
    """A predictor that outputs the true density must beat one that outputs a constant."""
    from codetrack.engine.losses import CodeTrackLoss

    torch.manual_seed(1)
    m, n = 4, 16
    support = torch.zeros(m, n)
    for j in range(m):
        support[j, torch.randperm(n)[:6]] = 1.0
    h = support / support.sum(dim=1, keepdim=True)
    mask = (torch.rand(3, n) < 0.25).float()

    density = ((support @ mask.t()).t() / support.sum(dim=1)).clamp(1e-4, 1 - 1e-4)
    base = {"reliability_rgb": torch.full((3, n), 0.9),
            "reliability_tir": torch.full((3, n), 0.9),
            "locator_scattered": torch.full((3, n), 0.01),
            "H": h, "H_support": support}

    loss = CodeTrackLoss(lambda_correct=0.0)
    good = loss._detect_loss({**base, "syndrome": density.unsqueeze(1)}, mask,
                             torch.zeros_like(mask))
    bad = loss._detect_loss({**base, "syndrome": torch.full((3, 1, m), 0.5)}, mask,
                            torch.zeros_like(mask))
    assert good["syndrome"] < bad["syndrome"]


# ------------------------------------------------- review round 2: parity shortcuts
def test_tanner_checks_do_not_contain_the_parity_reference():
    """P0-B: no self-reference.  ``checks`` must not receive the parity it is compared to.

    ``VisualSyndrome`` compares ``checks`` against ``parity_ref(parity)``.  If the parity
    were also summed into ``checks``, the discrepancy could be computed against itself.
    """
    model = CodeTrack(tiny_cfg())
    assert not hasattr(model.tanner, "parity_to_check"), \
        "the parity->check shortcut layer is back"

    b, n, d = 2, 16, 64
    variables = torch.randn(b, n, d)
    variables_tir = torch.randn(b, n, d)
    identity = torch.randn(b, 4, model.code_dim)
    parity_a = torch.randn(b, 4, model.code_dim)
    h = model.decoder.matrix()

    obs = 0.5 * (model.codebook.to_code(variables) + model.codebook.to_code(variables_tir))
    g1 = model.tanner(variables, identity, parity_a, variables_tir=variables_tir,
                      H=h, observation=obs)
    # perturbing the parity alone must not move the checks at all
    g2 = model.tanner(variables, identity, torch.randn_like(parity_a) * 10,
                      variables_tir=variables_tir, H=h, observation=obs)
    assert torch.allclose(g1["checks"], g2["checks"], atol=1e-6), \
        "checks still depend on the parity reference"


def test_assignment_temperature_is_bounded_and_learnable():
    """P0-C: ``tau`` must be learnable yet unable to leave ``[tau_min, tau_max]``."""
    model = CodeTrack(tiny_cfg(assign_temperature=0.1, tau_min=0.02, tau_max=0.5))
    cb = model.codebook
    tau = cb.temperature()
    assert 0.02 - 1e-6 <= float(tau) <= 0.5 + 1e-6
    assert abs(float(tau) - 0.1) < 0.02, "init should reproduce the reference temperature"
    assert cb.temperature_logit.requires_grad

    with torch.no_grad():                       # drive the parameter to the extremes
        cb.temperature_logit.fill_(-80.0)
    assert float(cb.temperature()) >= cb.tau_min - 1e-6
    with torch.no_grad():
        cb.temperature_logit.fill_(80.0)
    assert float(cb.temperature()) <= cb.tau_max + 1e-6


def test_assignment_ignores_low_reliability_tokens():
    """P0-D: a token the model distrusts must not steer the parity mixture.

    ``w`` is computed from the (possibly corrupted) received word, so without a
    reliability gate the "redundancy" would be derived from the word it must correct.
    """
    torch.manual_seed(2)
    cb = CodeTrack(tiny_cfg()).codebook
    identity = torch.randn(1, 4, cb.code_dim)
    variables = torch.randn(1, 16, cb.code_dim)

    # Compare the SAME token under both reliabilities, against two different observations.
    # A distrusted token (r=0) is replaced by its neighbourhood mean, so it must react to a
    # change in the observation *less* than a trusted one (r=1) does.
    obs_a = variables.clone()
    obs_b = variables.clone()
    obs_b[:, :8] = torch.randn(1, 8, cb.code_dim) * 50     # scramble 8 of 16 positions

    trust = torch.ones(1, 16)
    distrust = torch.zeros(1, 16)
    t_shift = (cb.assignment(obs_a, identity, reliability=trust)[:, :8]
               - cb.assignment(obs_b, identity, reliability=trust)[:, :8]).abs().max()
    d_shift = (cb.assignment(obs_a, identity, reliability=distrust)[:, :8]
               - cb.assignment(obs_b, identity, reliability=distrust)[:, :8]).abs().max()

    assert float(t_shift) > 1e-3, "sanity: scrambling should move a trusted token"
    assert float(d_shift) < float(t_shift), \
        "a distrusted token steers the assignment as strongly as a trusted one"


# ------------------------------------------------- review round 2: FPN clean bypass
def test_fpn_taps_are_masked_with_the_same_corruption():
    """P0-E: the FPN bypass must be structurally closed, not merely gated.

    The taps come from inside the backbone, i.e. from *before* corruption.  If they are
    left clean, ``L_track`` can learn to ignore the whole correction branch.
    """
    torch.manual_seed(3)
    model = CodeTrack(tiny_cfg(use_fpn=True))
    batch = dummy_batch(model, b=2)
    corruption = {"enabled": True, "token": ["tok_random_erase"], "ratio": 0.5,
                  "severity": 0.4, "target": "both"}

    out = model(**batch, corruption=corruption)
    mask = out["token_mask_rgb"]
    assert float(mask.sum()) > 0, "test setup: no corruption was applied"

    lens_z = int(model.backbone.pos_embed_z.shape[1])
    # recompute from the corrupted-aware features the model actually used
    feats = model._apply_tap_corruption(
        model.backbone((batch["template_rgb"], batch["search_rgb"]),
                       (batch["template_tir"], batch["search_tir"]), return_inter=True),
        mask, out["token_mask_tir"])
    tap = feats["inter_r"]["block0"]
    search = tap[:, lens_z:]
    erased = search[mask > 0.5]
    assert erased.numel() > 0 and float(erased.abs().max()) == 0.0, \
        "erased positions still carry clean FPN features"

    # the *unmasked* taps really do still contain the erased content -- otherwise this
    # test would pass trivially because the backbone never had anything there
    clean_feats = model.backbone((batch["template_rgb"], batch["search_rgb"]),
                                 (batch["template_tir"], batch["search_tir"]),
                                 return_inter=True)
    clean_search = clean_feats["inter_r"]["block0"][:, lens_z:]
    assert float(clean_search[mask > 0.5].abs().max()) > 0.0


# ------------------------------------------------- review round 2: L_correct sub-terms
def test_correct_loss_rejects_passing_the_corrupted_input_through():
    """P0-F: ``L_gain`` must punish "corrected == corrupted".

    A decoder that returns its own input unchanged has zero ``L_correct`` under any
    absolute metric; the relative margin is what makes that a non-solution.
    """
    from codetrack.engine.losses import CodeTrackLoss

    torch.manual_seed(4)
    n, dim, m = 16, 32, 4
    clean = {"rgb": torch.randn(1, n, dim), "tir": torch.randn(1, n, dim)}
    corrupted = {"rgb": clean["rgb"] + 1.0, "tir": clean["tir"] + 1.0}
    mask = torch.zeros(1, n)
    mask[:, :8] = 1.0

    common = {"token_mask_rgb": mask, "token_mask_tir": mask,
              "identity_tokens": torch.randn(1, 4, 32),
              "corrupted_rgb": corrupted["rgb"], "corrupted_tir": corrupted["tir"],
              "clean_tokens": clean}

    loss = CodeTrackLoss(lambda_detect=0.0, lambda_identity=0.0, lambda_preserve=0.0,
                         lambda_gain=1.0, gain_beta=0.9)
    parts = loss({**common, "corrected_rgb": corrupted["rgb"].clone(),
                  "corrected_tir": corrupted["tir"].clone()}, None)
    # corrected == corrupted -> error_after == error_before -> gain = 0.1 * before
    assert float(parts["gain"]) > 0

    fixed = {**common,
             "corrected_rgb": clean["rgb"].clone(), "corrected_tir": clean["tir"].clone()}
    assert float(loss(fixed, None)["gain"]) == pytest.approx(0.0, abs=1e-6)


def test_preserve_term_enters_the_total():
    """``preserve`` used to be computed and then dropped on the floor."""
    from codetrack.engine.losses import CodeTrackLoss

    torch.manual_seed(5)
    n, dim = 16, 32
    clean = {"rgb": torch.randn(1, n, dim), "tir": torch.randn(1, n, dim)}
    mask = torch.zeros(1, n)
    mask[:, :4] = 1.0
    base = {"token_mask_rgb": mask, "token_mask_tir": mask,
            "identity_tokens": torch.randn(1, 4, 32),
            "corrupted_rgb": clean["rgb"].clone(), "corrupted_tir": clean["tir"].clone(),
            "clean_tokens": clean,
            "corrected_rgb": clean["rgb"] + 0.5, "corrected_tir": clean["tir"] + 0.5}

    off = CodeTrackLoss(lambda_detect=0.0, lambda_identity=0.0, lambda_preserve=0.0,
                        lambda_gain=0.0)(base, None)
    on = CodeTrackLoss(lambda_detect=0.0, lambda_identity=0.0, lambda_preserve=0.5,
                       lambda_gain=0.0)(base, None)
    assert float(on["preserve"]) > 0, "the healthy-token error is not being measured"
    assert float(on["loss"]) > float(off["loss"]), "preserve is computed but not applied"
    assert torch.isclose(on["loss"] - off["loss"],
                         on["preserve"] * 0.5), "preserve is not weighted as configured"


# ------------------------------------------------------- metric correctness
def test_chance_level_is_a_recall_not_a_hit_count():
    """``k * n_pos / n`` is the expected *hit count*, not recall -- it exceeds 1.

    With 256 tokens, 20% corrupted and k=5, that formula returns ~1.8, so a perfect
    localizer (recall 1.0) would look *worse* than chance and the metric is unusable.
    """
    from codetrack.engine.evaluator import chance_level, recall_at_k

    n, ratio, k = 256, 0.2, 5
    n_pos = int(n * ratio)
    labels = np.zeros(n, dtype=bool)
    labels[:n_pos] = True
    rng = np.random.default_rng(0)

    # a random scorer must land at chance, and chance must stay <= 1
    recalls = [recall_at_k(rng.random(n), labels, k) for _ in range(200)]
    assert np.mean(recalls) == pytest.approx(chance_level(k, n, n_pos), abs=0.05)
    assert chance_level(k, n, n_pos) <= 1.0

    # A perfect scorer reaches the *ceiling* of recall@k, which is min(k, n_pos)/n_pos --
    # with 51 corrupted tokens and k=5 that is 0.098, five times chance.  This ceiling
    # is inherent to recall@k and is why the chance baseline, not 1.0, is the number to
    # compare against.
    ceiling = min(k, n_pos) / n_pos
    scores = labels.astype(np.float64) * 10.0 + rng.random(n)
    assert recall_at_k(scores, labels, k) == pytest.approx(ceiling)
    assert recall_at_k(scores, labels, k) == pytest.approx(
        5 * chance_level(k, n, n_pos), rel=0.02)


def test_locator_recall_is_computed_per_frame():
    """A global top-k over the whole corpus measures frame difficulty, not localization.

    Frame A has all its corrupted tokens scored high, frame B has all of them scored low.
    A global ranking would rank A first no matter how good the model is on B; a per-frame
    ranking scores both by the model's actual ordering.
    """
    from codetrack.engine.evaluator import recall_at_k

    labels = np.zeros((2, 8), dtype=bool)
    labels[0, :2] = True          # frame A: 2 corrupted, ranked top-2  -> recall 1.0
    labels[1, 6:] = True          # frame B: 2 corrupted, ranked LAST-2 -> recall 0.0
    scores = np.zeros((2, 8))
    scores[0, :2] = 10.0
    scores[1, :6] = 10.0

    assert recall_at_k(scores, labels, 2) == pytest.approx(0.5)


# ------------------------------------------------- syndrome output dead point
def test_syndrome_head_starts_at_the_expected_density():
    """P0-G: a sigmoid head at bias 0 is stuck at 0.5 and the soft target never moves it.

    Symptom before the fix: ``detect_syndrome`` sat at 0.694 -> 0.706 (= ln 2, the soft-BCE
    of a constant 0.5 prediction) for thousands of steps while the target sat at 0.2.  The
    sigmoid's gradient is not the problem -- starting 2.7 logits from the optimum is: the
    head must first unlearn its own initialisation before it can fit anything.

    The final bias is preset to logit(expected density), so the head starts *at* the answer
    and only has to learn the deviation.
    """
    model = CodeTrack(tiny_cfg())
    prior = 0.2
    syn = model.syndrome
    with torch.no_grad():
        out = syn(torch.randn(4, 4, syn.phi[0].in_features), torch.randn(4, 4, syn.parity_ref.in_features))
    got = float(out["syndrome"].mean())
    assert 0.05 < got < 0.45, f"syndrome starts at {got:.3f}, expected near the {prior} prior"

    # and it must be able to move AWAY from the prior, i.e. the prior is not a clamp
    bias = syn.sp2[-2].bias
    assert isinstance(bias, torch.nn.Parameter) and bias.requires_grad


def test_syndrome_raw_carries_an_explicit_discrepancy_scale():
    """``agg - ref`` between two LayerNorm-ed vectors is small; the RMS base term keeps a
    gradient path even when the learned projection shrinks towards zero."""
    model = CodeTrack(tiny_cfg())
    syn = model.syndrome
    d = syn.phi[0].out_features
    checks = torch.randn(3, 4, d)
    parity = torch.randn(3, 4, syn.parity_ref.in_features)

    out = syn(checks, parity)
    base = (out["check_agg"] - out["check_ref"]).pow(2).mean(-1).sqrt()
    ratio = (out["syndrome_raw"] / base.clamp(min=1e-6))
    # d_scale is learnable and starts at 1, so raw ~= RMS(delta) + d(delta); the ratio must
    # stay finite and positive rather than collapsing to 0
    assert torch.isfinite(ratio).all() and float(ratio.min()) > 1e-3
