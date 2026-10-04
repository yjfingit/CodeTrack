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
# tools/ holds locality_geometry, whose spatial_alignment is the measurement used below
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

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
    # The MLP width is solved from the BP message-module parameter count.
    assert 0.98 < mlp_params / bp_params < 1.02, \
        f"baseline not parameter-matched: mlp={mlp_params} bp={bp_params}"

    out = mlp.eval()(**dummy_batch(mlp))
    assert out["corrected_rgb"].shape == (2, mlp.num_variables, 64)


def test_spatial_mixer_baseline_is_parameter_matched_and_grid_local():
    """P5: the primary control is a *local spatial* mixer without parity or syndrome.

    It must (a) be usable and parameter-matched against BP, (b) mix only a limited neighbourhood
    -- otherwise a conv over the grid would silently be a global mixer -- and (c) not depend on
    the parity/syndrome inputs at all, since those are what the Tanner path claims to add.
    """
    spatial_cfg = tiny_cfg()
    spatial_cfg["model"]["decoder_mode"] = "spatial"
    spatial = CodeTrack(spatial_cfg)
    bp = CodeTrack(tiny_cfg())

    assert spatial.decoder.spatial_conv is not None
    assert len(spatial.decoder.spatial_conv) == 2          # one per iteration
    assert spatial.decoder.grid ** 2 == spatial.num_variables
    bp_params = sum(p.numel() for p in bp.decoder.parameters())
    spatial_params = sum(p.numel() for p in spatial.decoder.parameters())
    assert 0.98 < spatial_params / bp_params < 1.02, \
        f"spatial arm not parameter-matched: {spatial_params} vs {bp_params}"

    decoder = spatial.decoder.eval()
    b, n, d = 1, decoder.num_variables, 64
    grid = decoder.grid
    index = torch.stack([torch.arange(8)])
    gates = (torch.ones(b, 8), torch.ones(b, 8))
    parity = torch.randn(b, decoder.num_parity, 32)
    syndrome = torch.rand(b, 1, decoder.num_parity)
    reliability = torch.zeros(b, n)

    def run(rgb, tir, parity_in, syndrome_in):
        return decoder(rgb, tir, parity_in, syndrome_in, reliability, reliability,
                       gate_rgb=gates[0], gate_tir=gates[1], node_index=index)

    rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
    baseline = run(rgb, tir, parity, syndrome)
    # (c) parity and syndrome are not inputs of this path
    changed = run(rgb, tir, torch.randn(b, decoder.num_parity, 32),
                  torch.rand(b, 1, decoder.num_parity))
    assert torch.equal(baseline["corrected_rgb"], changed["corrected_rgb"])

    # (b) locality: with two rounds information travels at most two grid steps
    perturbed = rgb.clone()
    perturbed[0, 0] += 5.0
    after = run(perturbed, tir, parity, syndrome)
    difference = (after["corrected_rgb"] - baseline["corrected_rgb"]).abs().amax(dim=-1)[0]
    for token in range(n):
        row, column = divmod(token, grid)
        distance = max(row, column)
        if distance >= 3:
            assert float(difference[token]) == 0.0, \
                f"token {token} at Chebyshev distance {distance} changed: not a local mixer"
    assert float(difference[0]) > 0.0
    assert float(difference[0]) > float(difference[grid + 1])


def test_identity_residual_output_starts_as_an_exact_passthrough():
    """P2: the new parameterisation must begin at the identity, in both decoder modes.

    ``residual_out`` is zero-initialised, so the very first step is exactly zero, the output is
    not normalised, and the corrected tokens equal the inputs bit for bit.  That is what makes
    the identity-residual arm comparable to the ``off`` arm at initialisation.
    """
    for mode in ("bp", "mlp"):
        cfg = tiny_cfg()
        cfg["model"]["decoder_mode"] = mode
        cfg["model"]["decoder_output"] = "identity_residual"
        decoder = CodeTrack(cfg).decoder.eval()
        b, n, d = 2, decoder.num_variables, 64
        rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
        index = torch.stack([torch.randperm(n)[:8] for _ in range(b)])
        out = decoder(rgb, tir, torch.randn(b, decoder.num_parity, 32),
                      torch.rand(b, 1, decoder.num_parity), torch.rand(b, n), torch.rand(b, n),
                      gate_rgb=torch.rand(b, 8), gate_tir=torch.rand(b, 8), node_index=index)

        assert torch.equal(out["corrected_rgb"], rgb), f"{mode}: output is not the identity"
        assert torch.equal(out["corrected_tir"], tir), f"{mode}: output is not the identity"
        assert torch.count_nonzero(out["residual_rgb"]) == 0
        # no output LayerNorm exists in this mode, so the norm-only cell coincides with the
        # input: the four-cell probe degenerates and cannot be read as a baseline
        assert torch.equal(out["norm_only_rgb"], rgb)


def test_residual_clip_bounds_every_step():
    cfg = tiny_cfg()
    cfg["model"]["decoder_output"] = "identity_residual"
    cfg["model"]["residual_clip"] = 0.5
    decoder = CodeTrack(cfg).decoder.eval()
    torch.nn.init.constant_(decoder.residual_out.bias, 3.0)     # force a large step

    step = decoder._step(torch.randn(2, decoder.num_variables, 64))
    norms = step.norm(dim=-1)

    assert float(norms.max()) <= 0.5 + 1e-5
    assert float(norms.min()) > 0.0                              # clipping, not zeroing


def test_residual_clip_zero_is_off():
    cfg = tiny_cfg()
    cfg["model"]["decoder_output"] = "identity_residual"
    decoder = CodeTrack(cfg).decoder.eval()
    torch.nn.init.constant_(decoder.residual_out.bias, 3.0)

    step = decoder._step(torch.zeros(2, decoder.num_variables, 64))

    assert float(step.norm(dim=-1).min()) > 1.0                  # untouched by the clip


def test_gate_always_one_removes_the_learned_severity_gate():
    cfg = tiny_cfg()
    decoder = CodeTrack(cfg).decoder.eval()                      # default post_norm mode
    b, n, d = 1, decoder.num_variables, 64
    rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
    index = torch.stack([torch.randperm(n)[:8]])
    args = (torch.randn(b, decoder.num_parity, 32), torch.rand(b, 1, decoder.num_parity),
            torch.rand(b, n), torch.rand(b, n))
    ones = torch.ones(b, 8)

    explicit = decoder(rgb, tir, *args, gate_rgb=ones, gate_tir=ones, node_index=index)
    decoder.gate_always_one = True
    override = decoder(rgb, tir, *args, gate_rgb=torch.zeros(b, 8),
                       gate_tir=torch.zeros(b, 8), node_index=index)

    assert torch.equal(explicit["pre_norm_rgb"], override["pre_norm_rgb"])
    assert torch.equal(explicit["pre_norm_tir"], override["pre_norm_tir"])


def test_default_output_mode_still_applies_the_output_norm():
    cfg = tiny_cfg()
    decoder = CodeTrack(cfg).decoder.eval()
    b, n, d = 1, decoder.num_variables, 64
    rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
    index = torch.stack([torch.randperm(n)[:8]])
    out = decoder(rgb, tir, torch.randn(b, decoder.num_parity, 32),
                  torch.rand(b, 1, decoder.num_parity), torch.rand(b, n), torch.rand(b, n),
                  gate_rgb=torch.ones(b, 8), gate_tir=torch.ones(b, 8), node_index=index)

    assert decoder.output_mode == "post_norm"
    assert not torch.allclose(out["corrected_rgb"], out["pre_norm_rgb"])
    assert decoder.residual_out is None and decoder.branch_norm is None


def _state_mode_args(decoder, batch: int = 1, seed: int = 0):
    """Fixed inputs for the state-mode tests, so only the loop semantics can differ."""
    generator = torch.Generator().manual_seed(seed)
    n, d, m = decoder.num_variables, 64, decoder.num_parity
    return dict(
        variables_rgb=torch.randn(batch, n, d, generator=generator),
        variables_tir=torch.randn(batch, n, d, generator=generator),
        parity=torch.randn(batch, m, 32, generator=generator),
        syndrome=torch.rand(batch, 1, m, generator=generator),
        reliability_rgb=torch.rand(batch, n, generator=generator),
        reliability_tir=torch.rand(batch, n, generator=generator),
        gate_rgb=torch.rand(batch, 8, generator=generator),
        gate_tir=torch.rand(batch, 8, generator=generator),
        node_index=torch.stack([torch.randperm(n, generator=generator)[:8]
                                for _ in range(batch)]),
    )


def test_recurrent_state_mode_is_the_default_and_reads_the_updated_state():
    """073c223 silently froze the branch input for every non-identity mode, so from round 2 on the
    message functions read the decoder's *input* instead of the state round 1 produced.  That is a
    change of function, not of numerics, and it shipped while the commit claimed the default path
    was untouched (docs/results.md 6.27).  The default must be the pre-073c223 semantics, and it is
    checked here against an independently written two-round loop rather than against itself."""
    cfg = tiny_cfg()
    decoder = CodeTrack(cfg).decoder.eval()
    assert decoder.state_mode == "recurrent"
    args = _state_mode_args(decoder)

    with torch.no_grad():
        out = decoder(**args)

        # Independent re-implementation of the pre-073c223 loop for the post_norm path.
        b = args["variables_rgb"].shape[0]
        v = torch.cat([args["variables_rgb"], args["variables_tir"]], dim=0)
        r = torch.cat([args["reliability_rgb"], args["reliability_tir"]], dim=0).unsqueeze(-1)
        gate = torch.cat([
            decoder._expand_gate(args["gate_rgb"], args["node_index"], b, v.shape[1],
                                 args["variables_rgb"]),
            decoder._expand_gate(args["gate_tir"], args["node_index"], b, v.shape[1],
                                 args["variables_tir"]),
        ], dim=0).unsqueeze(-1)
        h = decoder._h()
        parity_c = args["parity"].repeat(2, 1, 1)
        s_col = args["syndrome"].repeat(2, 1, 1).flatten(1).unsqueeze(-1)
        parity_ctx = decoder.parity_proj(parity_c)
        m_cv = torch.zeros_like(v)
        for _ in range(decoder.iterations):
            m_vc = torch.einsum("cv,bvd->bcd", h,
                                decoder.v_msg(torch.cat([v, r], dim=-1)))
            c_in = torch.cat([m_vc, parity_ctx, s_col.expand(-1, -1, 1)], dim=-1)
            m_cv = torch.einsum("cv,bcd->bvd", h, decoder.c_msg(c_in))
            delta = decoder._step(decoder.update(torch.cat([v, m_cv], dim=-1)))
            v = v + (1.0 - r) * gate * delta
        reference = decoder.out_norm(v)

    assert torch.equal(torch.cat([out["corrected_rgb"], out["corrected_tir"]]), reference), \
        "the default state mode no longer reproduces the pre-073c223 message-passing loop"


def test_held_state_mode_differs_only_from_the_second_round():
    """The break 073c223 introduced is localised: with one round there is no previous state to
    read, so the two modes must agree bit for bit.  With two rounds they must not -- otherwise the
    flag would be decorative and the equivalence tool would be measuring nothing."""
    cfg = tiny_cfg()
    decoder = CodeTrack(cfg).decoder.eval()

    for iterations, should_differ in ((1, False), (2, True)):
        decoder.iterations = iterations
        args = _state_mode_args(decoder)
        with torch.no_grad():
            decoder.state_mode = "recurrent"
            recurrent = decoder(**args)["corrected_rgb"]
            decoder.state_mode = "held"
            held = decoder(**args)["corrected_rgb"]

        if should_differ:
            assert not torch.allclose(recurrent, held), \
                f"iterations={iterations}: 'held' should change the second round's input"
        else:
            assert torch.equal(recurrent, held), \
                "with a single round the state mode cannot matter"


def test_unknown_state_mode_is_rejected():
    cfg = tiny_cfg()
    cfg["model"]["decoder_state_mode"] = "sideways"
    with pytest.raises(ValueError, match="decoder state_mode"):
        CodeTrack(cfg)


def test_explicit_mlp_width_overrides_the_auto_formula():
    """The stored arm in ``outputs/ab_mlp_corr`` was trained at 2048 hidden (the pre-fix
    auto formula) and cannot be rebuilt without naming the width.  ``mlp_hidden=0`` must stay
    the auto (parameter-matched) value so the baseline path is unchanged."""
    auto_cfg = tiny_cfg()
    auto_cfg["model"]["decoder_mode"] = "mlp"
    auto = CodeTrack(auto_cfg)
    auto_hidden = auto.decoder.mlp_blocks[0][0].out_features

    wide_cfg = tiny_cfg()
    wide_cfg["model"]["decoder_mode"] = "mlp"
    wide_cfg["model"]["mlp_hidden"] = auto_hidden * 2
    wide = CodeTrack(wide_cfg)

    assert wide.decoder.mlp_blocks[0][0].out_features == auto_hidden * 2
    assert (sum(p.numel() for p in wide.decoder.parameters())
            > sum(p.numel() for p in auto.decoder.parameters()))


def test_both_decoder_modes_apply_the_severity_gate_identically():
    """The MLP baseline must receive the same severity gate as the BP decoder.

    The review's ablation caveat was that the MLP arm had no severity gate, so a full-vs-MLP
    difference also measured the gate.  Both branches multiply the update by ``(1 - r) *
    gate``, which a full-grid zero gate makes falsifiable: with the gate closed the pre-norm
    output must be the untouched input in *both* modes.
    """
    for mode in ("bp", "mlp"):
        cfg = tiny_cfg()
        cfg["model"]["decoder_mode"] = mode
        decoder = CodeTrack(cfg).decoder.eval()
        b, n, d = 1, decoder.num_variables, 64
        rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
        index = torch.stack([torch.randperm(n)[:8]])
        empty = torch.zeros(b, n)

        closed = decoder(rgb, tir, torch.randn(b, decoder.num_parity, 32),
                         torch.rand(b, 1, decoder.num_parity), torch.rand(b, n),
                         torch.rand(b, n), gate_rgb=empty, gate_tir=empty,
                         node_index=index)
        open_ = decoder(rgb, tir, torch.randn(b, decoder.num_parity, 32),
                        torch.rand(b, 1, decoder.num_parity), torch.rand(b, n),
                        torch.rand(b, n), gate_rgb=torch.ones(b, n),
                        gate_tir=torch.ones(b, n), node_index=index)

        assert torch.equal(closed["pre_norm_rgb"], rgb), f"{mode}: closed gate moved tokens"
        assert torch.equal(closed["pre_norm_tir"], tir), f"{mode}: closed gate moved tokens"
        assert not torch.allclose(open_["pre_norm_rgb"], rgb), f"{mode}: open gate did nothing"


def test_graph_node_gate_is_scattered_to_its_variable_indices():
    model = CodeTrack(tiny_cfg())
    index = torch.tensor([[7, 1, 12, 3, 14, 0, 9, 5]])
    gate = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

    expanded = model.decoder._expand_gate(
        gate, index, batch=1, num_variables=model.num_variables,
        reference=torch.zeros(1, model.num_variables))
    expected = torch.ones(1, model.num_variables)
    expected.scatter_(1, index, gate)
    assert torch.equal(expanded, expected)
    assert torch.equal(expanded[0, index[0]], gate[0])


def test_zero_gate_preserves_pre_norm_tokens_and_reports_norm_effect():
    model = CodeTrack(tiny_cfg()).decoder.eval()
    b, n, d = 1, model.num_variables, 64
    rgb, tir = torch.randn(b, n, d), torch.randn(b, n, d)
    index = torch.stack([torch.randperm(n)[:8]])
    zeros = torch.zeros(b, n)
    out = model(rgb, tir, torch.randn(b, model.num_parity, 32),
                torch.rand(b, 1, model.num_parity), torch.rand(b, n), torch.rand(b, n),
                gate_rgb=zeros, gate_tir=zeros, node_index=index)

    assert torch.equal(out["pre_norm_rgb"], rgb)
    assert torch.equal(out["pre_norm_tir"], tir)
    assert torch.count_nonzero(out["residual_rgb"]) == 0
    assert torch.count_nonzero(out["residual_tir"]) == 0
    assert not torch.equal(out["corrected_rgb"], rgb)  # post-norm still changes the value


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
    assert isinstance(syn.sp2.bias, torch.nn.Parameter) and syn.sp2.bias.requires_grad

    # the logit is exposed so the loss can use BCEWithLogitsLoss instead of a hand-written
    # sigmoid->clamp->log; sigmoid(logit) must reproduce the sigmoid output exactly
    checks = torch.randn(2, 4, syn.phi[0].out_features)
    parity = torch.randn(2, 4, syn.parity_ref.in_features)
    o = syn(checks, parity)
    assert torch.allclose(o["syndrome_logits"].squeeze(1).sigmoid(), o["syndrome"].squeeze(1),
                          atol=1e-6)


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


# ------------------------------------------------- locality prior on H's support
def test_locality_window_widens_the_per_check_density_contrast():
    """The whole point of the prior, measured on H alone -- no network involved.

    A spatially coherent burst is invisible to a uniformly random H: every check sees an
    arbitrary fixed-size set, so the per-check densities all land near the corruption ratio
    (measured std/mean = 0.31).  Aligning H with the search grid widens that spread 4x, which
    is what gives the syndrome head something to rank.  Under *independent* erasure the
    prior must change nothing -- there is no spatial structure to align with.
    """
    from codetrack.models.decoder.bp import NeuralBPDecoder

    n, m, grid = 256, 16, 16
    # a 2-D block, matching the prior's geometry.  A 1-D run of 40 flattened indices is NOT a
    # spatial neighbourhood and would be matched by accident rather than by geometry.
    burst = torch.zeros(n, dtype=torch.bool)
    burst[(5 * grid + 5):(9 * grid + 9)] = True         # 4x4 = 16 cells
    rand = torch.zeros(n, dtype=torch.bool)
    rand[torch.randperm(n)[:16]] = True

    def contrast(window: int, mask: torch.Tensor, seed: int = 0) -> float:
        torch.manual_seed(seed)
        dec = NeuralBPDecoder(dim=8, num_variables=n, num_parity=m, links_per_check=24,
                              min_column_degree=2, locality_window=window,
                              generator=torch.Generator().manual_seed(seed))
        sup = dec.connectivity()
        deg = sup.sum(dim=1).clamp(min=1.0)
        q = (sup @ mask.float()) / deg
        return float(q.std() / q.mean())

    off = contrast(0, burst)
    on = contrast(5, burst)                             # half-width 2 => 5x5 cells
    assert on > 1.5 * off, f"locality prior did not widen the contrast: {off:.3f} -> {on:.3f}"

    # and it must be a no-op for spatially incoherent damage
    assert abs(contrast(0, rand, 1) - contrast(5, rand, 1)) < 0.15


def test_locality_prior_keeps_the_support_sparse_and_the_column_degree_valid():
    """The window restricts the candidate set, not the learning: every check still picks its
    own edges, keeps `min_column_degree`, and the support stays a fixed binary mask."""
    model = CodeTrack(tiny_cfg(h_locality_window=3, h_links_per_check=4,
                               h_free_edge_frac=0.25))
    sup = model.decoder.connectivity()
    assert set(sup.unique().tolist()) <= {0.0, 1.0}, "support must stay binary"
    assert float(sup.sum(dim=1).min()) > 0, "every check must watch something"
    assert int(sup.sum(dim=0).min()) >= 2, "min_column_degree must still hold"
    # the realised degree is links_per_check + free_edge_frac*links_per_check, plus whatever
    # the min_column_degree repair had to add -- it is never below the requested value
    assert float(sup.sum(dim=1).min()) >= 4
    assert float(sup.sum()) < model.num_parity * model.num_variables


def test_locality_window_zero_is_the_original_uniform_support():
    """``h_locality_window: 0`` must reproduce the pre-prior behaviour exactly, so the
    ablation in the paper is a flag rather than a different model."""
    from codetrack.models.decoder.bp import NeuralBPDecoder

    torch.manual_seed(0)
    a = CodeTrack(tiny_cfg(h_locality_window=0)).decoder.connectivity()
    torch.manual_seed(0)
    b = CodeTrack(tiny_cfg(h_locality_window=0)).decoder.connectivity()
    assert torch.equal(a, b)
    # and with a window the watched cells really are spatially tighter.  Compactness is the
    # mean pairwise distance between a check's cells, normalised by a uniform draw of the
    # same size: < 1 means more local than chance.  (Counting how many *pairs of checks* share
    # a variable is not a valid proxy here -- window overlap makes that non-monotonic, it
    # went 858 -> 672 -> 1016 across windows 0/8/16.)
    from tools.locality_geometry import spatial_alignment

    torch.manual_seed(0)
    uniform = NeuralBPDecoder(dim=8, num_variables=256, num_parity=16,
                              links_per_check=24, min_column_degree=2,
                              locality_window=0).connectivity()
    torch.manual_seed(0)
    windowed = NeuralBPDecoder(dim=8, num_variables=256, num_parity=16,
                               links_per_check=24, min_column_degree=2,
                               locality_window=5).connectivity()
    assert spatial_alignment(windowed, 16) < 0.95 * spatial_alignment(uniform, 16)


def test_degree_balanced_geometry_groups_have_identical_degree_distributions():
    from codetrack.models.decoder.bp import NeuralBPDecoder

    supports = []
    for window in (0, 5, 32):
        decoder = NeuralBPDecoder(
            dim=8, num_variables=256, num_parity=16, links_per_check=32,
            min_column_degree=2, locality_window=window, free_edge_frac=0.25,
            balance_degrees=True, generator=torch.Generator().manual_seed(0))
        supports.append(decoder.connectivity())

    row_degrees = [support.sum(dim=1).long().sort().values for support in supports]
    column_degrees = [support.sum(dim=0).long().sort().values for support in supports]
    assert all(torch.equal(row_degrees[0], values) for values in row_degrees[1:])
    assert all(torch.equal(column_degrees[0], values) for values in column_degrees[1:])
    assert torch.all(row_degrees[0] == 32)
    assert torch.all(column_degrees[0] == 2)


def test_obs_energy_bypass_is_off_by_default():
    """Under zero-erasure the observation energy *is* the density label (corr 1.000), so the
    syndrome head must not see it unless explicitly asked.  Any result claiming the Tanner
    structure produced the syndrome depends on this default."""
    assert CodeTrack(tiny_cfg()).syndrome.use_obs_energy is False
    assert CodeTrack(tiny_cfg(syndrome_use_obs_energy=True)).syndrome.use_obs_energy is True


def test_config_override_does_not_turn_off_into_false():
    """YAML 1.1 reads ``off`` as a boolean, so ``--override model.decoder_mode=off`` became
    ``False`` and the decoder fell through to the BP branch with ``parity_proj = None``.

    The crash looked unrelated to the ablation it was supposed to run, which is exactly the
    kind of failure that gets misread as "the ablation is unstable".
    """
    from codetrack.utils.config import load_config

    cfg = load_config("configs/default.yaml", ["model.decoder_mode=off"])
    assert cfg["model"]["decoder_mode"] == "off"

    # only true/false are booleans here; "yes"/"no"/"on"/"off" stay strings so that
    # decoder_mode=off and discrepancy=off are expressible at all
    for raw, want in (("true", True), ("false", False)):
        c = load_config("configs/default.yaml", [f"model.freeze_backbone={raw}"])
        assert c["model"]["freeze_backbone"] is want, raw
    for raw in ("yes", "no", "on", "off"):
        c = load_config("configs/default.yaml", [f"model.discrepancy={raw}"])
        assert c["model"]["discrepancy"] == raw, raw

    # numbers and lists survive
    c = load_config("configs/default.yaml", ["model.h_links_per_check=24",
                                             "model.return_stages=[2,5]"])
    assert c["model"]["h_links_per_check"] == 24
    assert list(c["model"]["return_stages"]) == [2, 5]


def test_decoder_off_is_a_true_passthrough():
    """``decoder_mode="off"`` must remove the branch, not merely un-penalise it.

    Zeroing the loss weights would leave the decoder running and still rewriting the tokens,
    so a "no decoder" arm would be a normally-trained decoder that nobody supervised.
    """
    model = CodeTrack(tiny_cfg(decoder_mode="off")).eval()
    assert model.decoder.out_norm is None
    assert model.decoder.v_msg is None and model.decoder.parity_proj is None

    b, n, d = 2, model.num_variables, model.dim
    out = model.decoder(torch.randn(b, n, d), torch.randn(b, n, d),
                        torch.randn(b, model.num_parity, model.code_dim),
                        torch.rand(b, 1, model.num_parity), torch.rand(b, n), torch.rand(b, n))
    assert torch.equal(out["corrected_rgb"], out["norm_only_rgb"])  # untouched
    assert float(out["residual_rgb"].abs().sum()) == 0.0


def test_identity_residual_zeroes_a_non_finite_step_and_counts_it():
    """An fp16 overflow in the un-normalised residual path must be a counted event, not a fatal
    device-side assert: one NaN in the update used to abort a whole training run."""
    cfg = tiny_cfg()
    cfg["model"]["decoder_output"] = "identity_residual"
    cfg["model"]["residual_clip"] = 1.0
    decoder = CodeTrack(cfg).decoder.eval()

    poisoned = torch.full((2, decoder.num_variables, 64), float("nan"))
    out = decoder._step(poisoned)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0
    assert decoder.nonfinite_steps == 1

    # a healthy step must not be touched or counted
    healthy = torch.randn(2, decoder.num_variables, 64)
    decoder._step(healthy)
    assert decoder.nonfinite_steps == 1


# --------------------------------------------------- capacity location (LoRA in the blocks)


def _tiny_backbone(lora=None):
    """A 2-block backbone with deterministic weights, so arms can be compared exactly."""
    from codetrack.models.backbone.vit import SharedViTBackbone

    return SharedViTBackbone(search_size=64, template_size=32, patch_size=16, embed_dim=64,
                             depth=2, num_heads=4, pretrain_img_size=32, return_stages=(0, 1),
                             freeze=True, tir_from_rgb=True, lora=lora)


def _tiny_pair_inputs():
    return ((torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64)),
            (torch.randn(1, 1, 32, 32), torch.randn(1, 1, 64, 64)))


def test_lora_is_an_exact_identity_at_init_and_leaves_the_rng_stream_alone():
    """Two properties the capacity-location comparison depends on.

    B is zero-initialised, so the adapter contributes nothing at step 0 and the adapted arm starts
    bit-identical to the frozen one.  And the adapter's own init must not draw from the *global*
    RNG: if it did, every CodeTrack module built after the backbone would receive different
    initial weights in the adapted arm, so F0/F1/A0/A1 would differ in two things instead of one.
    """
    from codetrack.models.backbone.build import lora_kwargs

    torch.manual_seed(1234)
    frozen = _tiny_backbone(None).eval()
    torch.manual_seed(1234)
    adapted = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 4, "alpha": 4.0})).eval()

    frozen_params = dict(frozen.named_parameters())
    adapted_params = dict(adapted.named_parameters())

    def adapted_name(name: str) -> str:
        # the wrapper nests the pretrained projection one level deeper; every other name is
        # unchanged, and ``load_ostrack_pretrained`` aliases the checkpoint keys back
        return (name.replace(".attn.qkv.", ".attn.qkv.base.")
                if ".attn.qkv." in name else name)

    assert all(adapted_name(name) in adapted_params for name in frozen_params), \
        "the adapter must add tensors and rename only the wrapped projection"
    for name, param in frozen_params.items():
        assert torch.equal(param, adapted_params[adapted_name(name)]), \
            f"{name} differs: the adapter init shifted the global RNG"

    rgb, tir = _tiny_pair_inputs()
    with torch.no_grad():
        a, b = frozen(rgb, tir), adapted(rgb, tir)
    for key, value in a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, b[key]), f"{key}: the adapter is not an identity at init"

    assert adapted.lora_parameter_count() > 0
    assert all(p.requires_grad for p in adapted._lora_parameters())
    # rank 4, 2 targets, 2 modalities, 2 blocks: 2 * 2 * (4*64 + 64*4) per block
    assert adapted.lora_parameter_count() == 2 * 2 * 2 * (4 * 64 + 64 * 4)


def test_lora_actually_moves_the_output_after_a_step():
    """The identity at init must not be the whole story: a gradient step has to change the
    backbone's output, otherwise the arm would be the frozen one with extra dead parameters."""
    from codetrack.models.backbone.build import lora_kwargs

    adapted = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 4, "alpha": 4.0})).eval()
    rgb, tir = _tiny_pair_inputs()
    before = adapted(rgb, tir)
    optimizer = torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad], lr=0.5)
    loss = sum(v.float().pow(2).mean() for v in before.values() if isinstance(v, torch.Tensor))
    loss.backward()
    optimizer.step()
    after = adapted(rgb, tir)
    assert not all(torch.equal(before[k], after[k]) for k in before
                   if isinstance(before[k], torch.Tensor))


def test_freeze_backbone_keeps_adapters_trainable():
    """`freeze_backbone()` must not silently turn the adapted arm into the frozen one: an arm whose
    adapter never receives a gradient is the exact confusion this whole experiment exists to
    avoid."""
    from codetrack.models.backbone.build import lora_kwargs

    adapted = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 4}))
    adapted.freeze_backbone()
    assert adapted.frozen is True
    lora = adapted._lora_parameters()
    assert lora and all(p.requires_grad for p in lora)
    pretrained_trainable = [name for name, p in adapted.named_parameters()
                            if p.requires_grad and name not in
                            {n for n, _ in ((f"x{i}", q) for i, q in enumerate(lora))}
                            and ".lora_a." not in name and ".lora_b." not in name]
    assert pretrained_trainable == []

    adapted.unfreeze_backbone()
    assert all(p.requires_grad for p in adapted.parameters())


def test_lora_rank_and_modality_routing_change_the_parameter_count():
    from codetrack.models.backbone.build import lora_kwargs

    per_modality = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 8}))
    shared = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 8, "per_modality": False}))
    wider = _tiny_backbone(lora_kwargs({"enabled": True, "rank": 16}))

    assert per_modality.lora_parameter_count() == 2 * shared.lora_parameter_count()
    assert wider.lora_parameter_count() == 2 * per_modality.lora_parameter_count()
    with pytest.raises(ValueError, match="per-modality"):
        per_modality.blocks[0].attn.qkv.delta(torch.randn(1, 4, 64), modality=None)


def test_lora_disabled_reproduces_the_frozen_backbone_parameter_names():
    """With the adapter off the module must not exist at all -- not merely be inactive -- so the
    state dict keys and the pretrained loading path stay exactly as they were."""
    plain = _tiny_backbone(None)
    names = set(dict(plain.named_parameters()))
    assert "blocks.0.attn.qkv.weight" in names
    assert not any(".lora_" in name for name in names)
    assert plain.lora_parameter_count() == 0
