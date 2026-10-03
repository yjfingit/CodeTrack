#!/usr/bin/env python
"""Is the locality prior doing geometric work, or is the target just easier?

Training runs answer this in ~20 minutes; the geometry answers it in seconds, so run this
first.  The question is whether a spatially contiguous burst produces *more contrast between
checks* once ``H`` is spatially aligned, or whether the ``pearson = 0.978`` at window 16 is
only because a 40-token burst happens to fit inside a 16-token window and therefore makes
the per-check density trivially bimodal.

For each window size we measure, on the real ``H_support``:

* ``d_std``            -- std of the per-check density across the 16 checks.  This is the
  target's spread; a head cannot do better than the target allows.
* ``check_contrast``   -- ``std(q) / mean(q)``, i.e. how bimodal the target is.  A bimodal
  target is an easy target no matter what the head does.
* ``spatial_alignment`` -- mean cosine between each check's watched cells and a spatial
  neighbourhood, versus the same for a random support.  This is the prior's own strength,
  measured independently of any training.

If ``check_contrast`` at window 16 is not materially higher than at window 128 (no prior),
then the gain came from an easier target and not from the geometry, and the number must not
be attributed to the prior.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-parity", type=int, default=16)
    ap.add_argument("--num-variables", type=int, default=256)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--burst", type=int, default=40)
    args = ap.parse_args()

    grid = int(round(args.num_variables ** 0.5))
    print(f"M={args.num_parity} N={args.num_variables} degree={args.degree} "
          f"grid={grid}x{grid} frames={args.frames} ratio={args.ratio}\n")

    for kind in ("burst", "random"):
        print(f"=== corruption: {kind} ===")
        print("%8s %10s %10s %14s %18s" % (
            "window", "d_std", "d_mean", "check_contrast", "spatial_align"))
        for w in (0, 8, 16, 32, 64, 128):
            dec = NeuralBPDecoder(dim=8, num_variables=args.num_variables,
                                  num_parity=args.num_parity,
                                  links_per_check=args.degree, min_column_degree=2,
                                  locality_window=w,
                                  generator=torch.Generator().manual_seed(0))
            sup = dec.connectivity()

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
            print("%8d %10.4f %10.4f %14.2f %18.3f" % (
                w, s, m, s / max(m, 1e-8), align))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
