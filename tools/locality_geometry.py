#!/usr/bin/env python
"""Is the locality prior doing geometric work, or is the target just easier?

This diagnostic measures check-density contrast, spatial alignment, and realized degrees
for the current support configuration. Density contrast alone does not establish that a
trained syndrome uses the pattern or that tracking improves.

For each window size we measure, on the real ``H_support``:

* ``d_std``            -- std of the per-check density across the 16 checks.  This is the
  target's spread; a head cannot do better than the target allows.
* ``check_contrast``   -- ``std(q) / mean(q)``, i.e. how bimodal the target is.  A bimodal
  target is an easy target no matter what the head does.
* ``spatial_alignment`` -- mean cosine between each check's watched cells and a spatial
  neighbourhood, versus the same for a random support.  This is the prior's own strength,
  measured independently of any training.

Interpret contrast together with the degree audit and matched-mask recovery and tracking
measurements. A contrast difference is a property of the label geometry, not model efficacy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.models.decoder.bp import NeuralBPDecoder  # noqa: E402


def densities(support: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``M`` per-check densities for one frame. ``support``: M x N {0,1}, ``mask``: N."""
    deg = support.sum(dim=1).clamp(min=1.0)
    return (support @ mask.float()) / deg


def spatial_alignment(support: torch.Tensor, grid: int) -> float:
    """Mean pairwise spatial proximity of a check's cells vs. a random draw of the same size.

    1.0 means every pair of watched cells is adjacent; ~1/N means the row is scattered.
    Returned relative to the random baseline so 1.0 = "no more local than chance".
    """
    pos = torch.arange(support.shape[1])
    rc = torch.stack([pos // grid, pos % grid], dim=1).float()

    def compactness(row: torch.Tensor) -> float:
        """Mean pairwise spatial distance among the cells this check watches.

        Small = the check looks at one contiguous patch; large = it is scattered.  For a
        row of k cells that fall uniformly at random over a grid, the expected value is
        proportional to the grid size, which is why the ratio to a random draw is reported
        rather than the raw distance.
        """
        sel = torch.nonzero(row).flatten()
        if sel.numel() < 2:
            return float("nan")
        d = (rc[sel][:, None, :] - rc[sel][None, :, :]).pow(2).sum(-1).sqrt()
        return float(d.mean())

    prior = np.mean([compactness(support[r]) for r in range(support.shape[0])])

    g = torch.Generator().manual_seed(0)
    rand_rows = []
    for r in range(support.shape[0]):
        k = int(support[r].sum())
        row = torch.zeros(support.shape[1])
        row[torch.randperm(support.shape[1], generator=g)[:k]] = 1.0
        rand_rows.append(row)
    rand = torch.stack(rand_rows)
    base = np.mean([compactness(rand[r]) for r in range(rand.shape[0])])
    return float(prior / max(base, 1e-8))


def make_support(args, window: int) -> torch.Tensor:
    dec = NeuralBPDecoder(dim=8, num_variables=args.num_variables,
                          num_parity=args.num_parity,
                          links_per_check=args.degree, min_column_degree=2,
                          locality_window=window, free_edge_frac=args.free_edge_frac,
                          balance_degrees=args.balance_degrees,
                          generator=torch.Generator().manual_seed(0))
    return dec.connectivity()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-parity", type=int, default=16)
    ap.add_argument("--num-variables", type=int, default=256)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--burst", type=int, default=40)
    ap.add_argument("--free-edge-frac", type=float, default=0.25)
    ap.add_argument("--balance-degrees", action="store_true")
    ap.add_argument("--windows", default="0,8,16,32,64,128")
    ap.add_argument("--audit-only", action="store_true",
                    help="report realized degrees and verify they match across windows")
    args = ap.parse_args()

    grid = int(round(args.num_variables ** 0.5))
    print(f"M={args.num_parity} N={args.num_variables} degree={args.degree} "
          f"grid={grid}x{grid} frames={args.frames} ratio={args.ratio}\n")
    windows = [int(value) for value in args.windows.split(",") if value.strip()]

    if args.audit_only:
        supports = {window: make_support(args, window) for window in windows}
        ref_window = windows[0]
        ref = supports[ref_window]
        ref_row_degree = ref.sum(dim=1).long().sort().values
        ref_col_degree = ref.sum(dim=0).long().sort().values
        for window, support in supports.items():
            row_degree = support.sum(dim=1).long()
            col_degree = support.sum(dim=0).long()
            print(f"window={window:>3} edges={int(support.sum())} "
                  f"row[min/mean/max]={row_degree.min()}/"
                  f"{row_degree.float().mean():.2f}/{row_degree.max()} "
                  f"col[min/mean/max]={col_degree.min()}/"
                  f"{col_degree.float().mean():.2f}/{col_degree.max()}")
            if not torch.equal(row_degree.sort().values, ref_row_degree):
                raise SystemExit(f"row-degree distribution differs at window={window}")
            if not torch.equal(col_degree.sort().values, ref_col_degree):
                raise SystemExit(f"column-degree distribution differs at window={window}")
        print("degree audit passed: all requested windows have matching row/column degrees")
        return 0

    for kind in ("burst", "random"):
        print(f"=== corruption: {kind} ===")
        print("%8s %10s %10s %14s %18s %10s %10s" % (
            "window", "d_std", "d_mean", "check_contrast", "spatial_align",
            "edges", "col[min,max]"))
        for w in windows:
            sup = make_support(args, w)
            col_degree = sup.sum(dim=0)

            k = int(round(args.ratio * args.num_variables)) if kind == "random" else args.burst
            k = min(k, args.num_variables)
            stds, means = [], []
            for _ in range(args.frames):
                if kind == "random":
                    idx = torch.randperm(args.num_variables)[:k]
                else:
                    start = int(torch.randint(0, max(1, args.num_variables - k), (1,)))
                    idx = (torch.arange(k) + start) % args.num_variables
                mask = torch.zeros(args.num_variables)
                mask[idx] = 1.0
                q = densities(sup, mask)
                stds.append(float(q.std()))
                means.append(float(q.mean()))
            m, s = float(np.mean(means)), float(np.mean(stds))
            align = spatial_alignment(sup, grid) if w else 1.0
            print("%8d %10.4f %10.4f %14.2f %18.3f %10d %5.0f,%5.0f" % (
                w, s, m, s / max(m, 1e-8), align, int(sup.sum()),
                float(col_degree.min()), float(col_degree.max())))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
