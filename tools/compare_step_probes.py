#!/usr/bin/env python
"""Compare step-size probes across arms: is the overshoot fixed, and does the gate matter?

The P2 arms differ only in how the decoder parameterises its output (and whether the learned
severity gate is used), so the *same* offline probe -- `tools/step_size_probe.py` -- is the
cheapest way to see whether an arm actually changed the update geometry before any tracking
comparison is believed.  This tool reads several probe artifacts and puts the decisive ratios
side by side:

* `applied/oracle` -- how far the applied step is from the ideal step.  > 1 is overshoot;
  an arm that fixes the step-size problem should be near 1.
* `L1(applied) - L1(zero)` -- negative means the update helps, positive means it hurts.
* `L1(gate-permuted) - L1(applied)` -- if permuting the gate changes nothing, the gate carries no
  per-token value information.
* `a*` zero fraction and `rho(gate, a*)` -- whether the direction is useful everywhere and
  whether the gate tracks the ideal step.

Usage::

    python tools/compare_step_probes.py --probe A=outputs/step_size_probe_p2_A.json \
        --probe D=outputs/step_size_probe_p2_D.json --out outputs/step_probe_table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load(path: Path) -> Dict[str, float]:
    """Probe metrics for one arm; a missing artifact reports why instead of crashing the table.

    During the P2 grid one arm can fail while the others finish, and a comparison table that
    raises on the missing file would throw away the arms that did complete.
    """
    if not path.exists():
        return {"error": f"no artifact at {path}"}
    data = json.loads(path.read_text())
    l1 = data.get("mean_l1", {})
    zero = l1.get("l1_zero")
    applied = l1.get("l1_applied")
    oracle = l1.get("l1_oracle")
    permuted = l1.get("l1_gate_permuted")
    return {
        "checkpoint": data.get("checkpoint"),
        "frames": data.get("frames"),
        "n_damaged_tokens": data.get("n_damaged_tokens"),
        "single_step": bool(data.get("single_step")),
        "l1_zero": zero,
        "l1_applied": applied,
        "l1_oracle": oracle,
        "l1_best_global": l1.get("l1_best_global"),
        "l1_gate_permuted": permuted,
        "overshoot_applied_over_oracle": (applied / oracle) if zero and oracle else None,
        "gain_vs_zero": (zero - applied) if (zero is not None and applied is not None) else None,
        "gate_permutation_effect": ((permuted - applied)
                                    if (permuted is not None and applied is not None) else None),
        "astar_zero_fraction": data.get("astar_zero_fraction"),
        "astar_median": data.get("astar_median"),
        "gate_median": data.get("gate_median"),
        "gate_selected_mean": data.get("gate_selected_mean"),
        "gate_astar_spearman": data.get("gate_astar_spearman"),
        "applied_ratio_to_d_median": data.get("applied_ratio_to_d_median"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable; the label becomes the table row")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows: Dict[str, Dict[str, float]] = {}
    for spec in args.probe:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--probe expects LABEL=PATH, got {spec!r}")
        rows[label] = load(Path(path))

    def value(row: Dict[str, float], key: str) -> str:
        item = row.get(key)
        if item is None:
            return "n/a"
        if isinstance(item, bool):
            return "yes" if item else "no"
        return f"{item:.4f}" if isinstance(item, float) else str(item)

    header = ("arm", "applied/oracle", "L1zero", "L1applied", "L1oracle", "gain_vs_zero",
              "gateperm-applied", "a*=0 frac", "rho(gate,a*)")
    print(f"{header[0]:<6} {header[1]:>15} {header[2]:>9} {header[3]:>10} {header[4]:>9} "
          f"{header[5]:>13} {header[6]:>17} {header[7]:>11} {header[8]:>13}")
    for label, row in rows.items():
        if "error" in row:
            print(f"{label:<6} {row['error']:>15}")
            continue
        print(f"{label:<6} {value(row, 'overshoot_applied_over_oracle'):>15} "
              f"{value(row, 'l1_zero'):>9} {value(row, 'l1_applied'):>10} "
              f"{value(row, 'l1_oracle'):>9} {value(row, 'gain_vs_zero'):>13} "
              f"{value(row, 'gate_permutation_effect'):>17} "
              f"{value(row, 'astar_zero_fraction'):>11} "
              f"{value(row, 'gate_astar_spearman'):>13}")
    print("\napplied/oracle ~ 1 and gain_vs_zero < 0 means the update is close to the oracle step "
          "and helps; gate permutation changing nothing means the severity gate is inert.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
