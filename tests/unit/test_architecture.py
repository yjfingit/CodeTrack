"""Architecture-property tests.

Each test locks one invariant that the review identified as violated in ``b530d9a``.
They run on a deliberately tiny backbone (2 blocks, dim 64) so the whole file finishes in
seconds on CPU and needs no pretrained checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
    """P0-3: locator_scattered must equal `s @ H` -- failing checks vote for variables."""
    model = CodeTrack(tiny_cfg())
    b, m, n, k = 2, model.num_parity, model.num_variables, model.num_graph_nodes
    syndrome = torch.rand(b, 1, m)
    identity_map = torch.rand(b, k)
    node_index = torch.stack([torch.randperm(n)[:k] for _ in range(b)])

    out = model.locator(syndrome, identity_map, node_index, n, H=model.decoder.matrix())
    expected = (syndrome.flatten(1) @ model.decoder.matrix()).clamp(0.0, 1.0)
    assert torch.allclose(out["locator_scattered"], expected, atol=1e-6)


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
