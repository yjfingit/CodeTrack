#!/usr/bin/env python
"""Measure what ``decoder_state_mode`` changes, and prove "recurrent" is the pre-073c223 path.

Why this tool exists
--------------------
Commit ``073c223`` replaced ``v`` with ``branch_input`` inside the decoder loop but refreshed
``branch_input`` only when ``output_mode == "identity_residual"``.  For every other mode that
*changed the function*: from round 2 on, the message and update functions read the decoder's
**input** instead of the state round 1 produced.  Nothing in the state dict changed, so
``load_checkpoint`` could not notice, and the commit was described as leaving the default
``post_norm`` path bit-identical.  It did not.

That matters because arms trained before ``073c223`` (``ab_full``, ``ab_nodec_*``, the shipped
reference) were subsequently evaluated by code running the other semantics, and because
``docs/results.md`` 6.10 attributed arm A's mismatch with ``ab_full`` to a shifted RNG stream
rather than to a code change.

What this tool does
-------------------
For each ``(mode, output_mode, iterations)`` cell it builds one decoder, copies its weights into
three instances and runs the same inputs through:

* ``recurrent`` -- the shipped default: round *l* reads the state round *l-1* produced;
* ``held`` -- round *l* reads the decoder input;
* ``parent`` -- an independent re-implementation of the pre-``073c223`` loop, written here on
  purpose so it is *not* the module's own code path (otherwise the check would be circular).

It then reports ``max |recurrent - parent|`` (must be 0 for every cell: this is the regression
guard) and ``max |recurrent - held|`` (the size of the semantic break that shipped).  With
``iterations = 1`` the two modes must agree bit for bit, which localises the difference to
multi-round state rather than to anything else.

Usage::

    python tools/decoder_state_equivalence.py --out outputs/decoder_state_equivalence.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.models.decoder import NeuralBPDecoder  # noqa: E402
from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401


def build(dim: int, iterations: int, mode: str, output_mode: str,
          state_mode: str, seed: int) -> NeuralBPDecoder:
    """One decoder with deterministic weights, so every instance starts from the same tensors."""
    torch.manual_seed(seed)
    decoder = NeuralBPDecoder(
        dim=dim, code_dim=dim // 2, num_parity=4, num_variables=16, iterations=iterations,
        links_per_check=4, min_column_degree=2, mode=mode, output_mode=output_mode,
        residual_clip=0.0, gate_always_one=False, state_mode=state_mode,
        locality_window=0, generator=torch.Generator().manual_seed(seed),
    )
    # Non-degenerate weights: a zero-initialised identity-residual projection would make every
    # cell report 0 for the uninteresting reason that the branch never fires.
    with torch.no_grad():
        for param in decoder.parameters():
            if param.dim() >= 2:
                param.add_(torch.randn_like(param) * 0.02)
        if decoder.residual_out is not None:
            decoder.residual_out.weight.add_(torch.randn_like(decoder.residual_out.weight) * 0.5)
            decoder.residual_out.bias.add_(torch.randn_like(decoder.residual_out.bias) * 0.5)
    return decoder.eval()


def inputs(dim: int, num_variables: int, num_parity: int, batch: int, seed: int):
    """Fixed inputs shared by every arm of one cell."""
    generator = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=generator)
    return {
        "variables_rgb": rand(batch, num_variables, dim),
        "variables_tir": rand(batch, num_variables, dim),
        "parity": rand(batch, num_parity, dim // 2),
        "syndrome": torch.rand(batch, 1, num_parity, generator=generator),
        "reliability_rgb": torch.rand(batch, num_variables, generator=generator),
        "reliability_tir": torch.rand(batch, num_variables, generator=generator),
        "gate_rgb": torch.rand(batch, 4, generator=generator),
        "gate_tir": torch.rand(batch, 4, generator=generator),
        "node_index": torch.stack([torch.randperm(num_variables, generator=generator)[:4]
                                   for _ in range(batch)]),
    }


def parent_forward(decoder: NeuralBPDecoder, args: Dict[str, torch.Tensor],
                   iterations: int) -> torch.Tensor:
    """The pre-``073c223`` loop, re-implemented here instead of called.

    ``3ef5083`` ran ``v_msg`` and ``update`` on ``v`` itself, so every round read the state the
    previous round had written.  Only the ``post_norm`` output is reproduced: the parent commit
    had no ``identity_residual`` mode at all, and the norm-only/pre-norm views are not what the
    semantic question is about.
    """
    with torch.no_grad():
        b = args["variables_rgb"].shape[0]
        v = torch.cat([args["variables_rgb"], args["variables_tir"]], dim=0)
        r = torch.cat([args["reliability_rgb"], args["reliability_tir"]], dim=0).unsqueeze(-1)
        gate_r = decoder._expand_gate(args["gate_rgb"], args["node_index"], b, v.shape[1],
                                      args["variables_rgb"])
        gate_t = decoder._expand_gate(args["gate_tir"], args["node_index"], b, v.shape[1],
                                      args["variables_tir"])
        gate_full = torch.cat([gate_r, gate_t], dim=0).unsqueeze(-1)

        if decoder.mode in ("mlp", "spatial"):
            for round_index, block in enumerate(decoder.mlp_blocks):
                features = v
                if decoder.mode == "spatial":
                    grid = decoder.grid
                    maps = features.transpose(1, 2).reshape(-1, decoder.dim, grid, grid)
                    mixed = decoder.spatial_conv[round_index](maps)
                    features = features + mixed.reshape(-1, decoder.dim,
                                                        grid * grid).transpose(1, 2)
                delta = decoder._step(block(torch.cat([features, r], dim=-1)))
                v = v + (1.0 - r) * gate_full * delta
            return decoder.out_norm(v)

        h = decoder._h()
        parity_c = (args["parity"].repeat(2, 1, 1)
                    if args["parity"].shape[0] == b else args["parity"])
        syndrome = args["syndrome"]
        s = syndrome.repeat(2, 1, 1) if syndrome.shape[0] == b else syndrome
        parity_ctx = decoder.parity_proj(parity_c)
        s_col = s.flatten(1).unsqueeze(-1)
        m_cv = torch.zeros_like(v)
        for _ in range(iterations):
            m_vc = torch.einsum("cv,bvd->bcd", h, decoder.v_msg(torch.cat([v, r], dim=-1)))
            c_in = torch.cat([m_vc, parity_ctx, s_col.expand(-1, -1, 1)], dim=-1)
            m_cv = torch.einsum("cv,bcd->bvd", h, decoder.c_msg(c_in))
            delta = decoder._step(decoder.update(torch.cat([v, m_cv], dim=-1)))
            v = v + (1.0 - r) * gate_full * delta
        return decoder.out_norm(v)


def run_cell(mode: str, output_mode: str, iterations: int, dim: int = 64,
             seed: int = 0) -> Dict[str, object]:
    """One ``(mode, output_mode, iterations)`` cell."""
    base = build(dim, iterations, mode, output_mode, "recurrent", seed)
    held = copy.deepcopy(base)
    held.state_mode = "held"
    args = inputs(dim, base.num_variables, base.num_parity, 2, seed + 1)

    with torch.no_grad():
        out_recurrent = base(**args)
        out_held = held(**args)

    row: Dict[str, object] = {"mode": mode, "output_mode": output_mode,
                              "iterations": iterations}
    for key in ("corrected_rgb", "corrected_tir"):
        row[f"max_abs_recurrent_vs_held_{key}"] = float(
            (out_recurrent[key] - out_held[key]).abs().max())
    row["held_differs"] = bool(row["max_abs_recurrent_vs_held_corrected_rgb"] > 0.0)

    if output_mode == "post_norm" and mode in ("bp", "mlp", "spatial"):
        with torch.no_grad():
            reference = parent_forward(base, args, iterations)
        row["max_abs_recurrent_vs_parent_corrected"] = float(
            (torch.cat([out_recurrent["corrected_rgb"], out_recurrent["corrected_tir"]])
             - reference).abs().max())
        row["recurrent_matches_parent"] = bool(
            row["max_abs_recurrent_vs_parent_corrected"] == 0.0)
    else:
        row["max_abs_recurrent_vs_parent_corrected"] = None
        row["recurrent_matches_parent"] = None

    if iterations == 1:
        row["single_round_modes_identical"] = bool(
            not row["held_differs"])
    else:
        row["single_round_modes_identical"] = None
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/decoder_state_equivalence.json")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cells: List[Dict[str, object]] = []
    for mode in ("bp", "mlp", "spatial"):
        for output_mode in ("post_norm", "identity_residual"):
            for iterations in (1, 2, 3):
                cells.append(run_cell(mode, output_mode, iterations, args.dim, args.seed))

    failures = [cell for cell in cells
                if cell["recurrent_matches_parent"] is False
                or (cell["iterations"] == 1 and cell["held_differs"])]
    report = {
        "note": ("'recurrent' is the pre-073c223 semantics and must match the independent parent "
                 "loop exactly; 'held' is what 073c223 silently applied to post_norm/mlp/spatial. "
                 "iterations=1 must be mode-independent."),
        "dim": args.dim,
        "seed": args.seed,
        "cells": cells,
        "n_cells": len(cells),
        "n_failures": len(failures),
        "failures": failures,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{'mode':<8} {'output_mode':<18} {'it':>2} {'|rec-held|':>12} "
          f"{'|rec-parent|':>13} {'held differs':>13}")
    for cell in cells:
        parent = cell["max_abs_recurrent_vs_parent_corrected"]
        print(f"{cell['mode']:<8} {cell['output_mode']:<18} {cell['iterations']:>2} "
              f"{cell['max_abs_recurrent_vs_held_corrected_rgb']:>12.6g} "
              f"{'n/a' if parent is None else f'{parent:>13.6g}'} "
              f"{str(cell['held_differs']):>13}")
    print(f"\n{len(cells)} cells, {len(failures)} contract failures -> {out}")
    if failures:
        print("FAIL: 'recurrent' must reproduce the pre-073c223 loop bit for bit, and the state "
              "mode must be irrelevant at iterations=1.")
        return 1
    print("OK: recurrent == parent on every post_norm cell; state mode is a no-op at 1 round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
