#!/usr/bin/env python
"""Select a stratified validation split of LasHeR sequences.

The earlier evaluations used the first N lines of ``testingsetList.txt``.  That is a
convenience sample: it over-represents whichever challenges happen to come first
alphabetically, and it cannot say anything about thermal crossover or low illumination,
which are exactly the conditions the corruption protocol is meant to stress.

This tool assigns every sequence to one challenge stratum from LasHeR's own attribute
annotations (``AttriSeqsTxt/``, order given by ``Attributes_order.txt``) and samples a
fixed number per stratum with a seeded RNG.  The selection is reproducible: the same
``--seed`` and the same attribute files always produce the same split.

Strata are mutually exclusive and assigned in this priority order (LasHeR sequences carry
several attributes, so an order is unavoidable and is recorded in the manifest):

1. ``low_illumination``  -- ``LI`` (RGB low light)
2. ``thermal_crossover`` -- ``TC`` and not ``LI`` (TIR loses target/background contrast)
3. ``total_occlusion``   -- ``TO`` and not ``LI``/``TC``
4. ``partial_occlusion`` -- ``PO`` without ``NO`` and not the above
5. ``unoccluded``        -- everything else (the easy majority)

Usage::

    python tools/select_validation_sequences.py --out outputs/validation_split_v1 --n 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

STRATA = ("low_illumination", "thermal_crossover", "total_occlusion",
          "partial_occlusion", "unoccluded")


def read_attributes(root: Path, sequence: str, order: Sequence[str]) -> Dict[str, int]:
    path = root / "AttriSeqsTxt" / f"{sequence}.txt"
    if not path.exists():
        return {}
    line = path.read_text().strip().splitlines()[0]
    values = [int(value) for value in line.replace("\t", ",").split(",") if value != ""]
    return {name: value for name, value in zip(order, values[:len(order)])}


def assign_stratum(attributes: Dict[str, int]) -> str:
    if attributes.get("LI") == 1:
        return "low_illumination"
    if attributes.get("TC") == 1:
        return "thermal_crossover"
    if attributes.get("TO") == 1:
        return "total_occlusion"
    if attributes.get("PO") == 1 and attributes.get("NO") != 1:
        return "partial_occlusion"
    return "unoccluded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset",
                        choices=["testingset", "trainingset"])
    parser.add_argument("--out", required=True, help="directory for sequences.txt/manifest")
    parser.add_argument("--n", type=int, default=60, help="total sequences to select")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", default=None,
                        help="existing sequences.txt to keep, so a larger split is a strict "
                             "superset of a smaller one and the two remain comparable on the "
                             "shared sequences")
    args = parser.parse_args()

    root = Path(args.root)
    order = [name.strip() for name in (root / "Attributes_order.txt").read_text().strip()
             .split(",") if name.strip()]
    list_name = ("testingsetList.txt" if args.subset.startswith("test")
                 else "trainingsetList.txt")
    available = [line.strip() for line in (root / list_name).read_text().splitlines()
                 if line.strip()]

    by_stratum: Dict[str, List[str]] = {name: [] for name in STRATA}
    attributes_of: Dict[str, Dict[str, int]] = {}
    for sequence in available:
        attributes = read_attributes(root, sequence, order)
        if not attributes:
            continue
        attributes_of[sequence] = attributes
        by_stratum[assign_stratum(attributes)].append(sequence)
    for name in STRATA:
        by_stratum[name] = sorted(by_stratum[name])

    total = sum(len(value) for value in by_stratum.values())
    if total < args.n:
        print(f"only {total} sequences have attributes; selecting all of them")
    quota = max(1, args.n // len(STRATA)) if args.n else 0
    rng = np.random.default_rng(args.seed)

    selected: List[str] = []
    chosen: Dict[str, List[str]] = {name: [] for name in STRATA}
    if args.include:
        # keep an earlier split verbatim: it becomes the first quota of its own stratum
        keep = [line.strip() for line in Path(args.include).read_text().splitlines()
                if line.strip()]
        for sequence in keep:
            if sequence in attributes_of:
                chosen[assign_stratum(attributes_of[sequence])].append(sequence)
                selected.append(sequence)
    for name in STRATA:
        candidates = [s for s in by_stratum[name] if s not in set(chosen[name])]
        take = min(max(0, quota - len(chosen[name])), len(candidates))
        if take:
            picks = rng.permutation(len(candidates))[:take]
            chosen[name].extend(candidates[i] for i in sorted(picks))
            selected.extend(candidates[i] for i in sorted(picks))

    # Fill the remainder round-robin over the strata so the balance is kept even when one
    # stratum is small (low illumination has only ~50 testingset sequences).
    leftovers = {name: [s for s in by_stratum[name] if s not in set(chosen[name])]
                 for name in STRATA}
    for name in STRATA:
        rng.shuffle(leftovers[name])
    while len(selected) < min(args.n, total):
        progressed = False
        for name in STRATA:
            if len(selected) >= min(args.n, total):
                break
            if leftovers[name]:
                chosen[name].append(leftovers[name].pop())
                selected.append(chosen[name][-1])
                progressed = True
        if not progressed:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequences.txt").write_text("\n".join(selected) + "\n")
    manifest = {
        "root": str(root),
        "subset": args.subset,
        "included_from": args.include,
        "seed": args.seed,
        "requested": args.n,
        "selected": len(selected),
        "attribute_order": order,
        "strata_priority": list(STRATA),
        "stratum_counts_available": {name: len(by_stratum[name]) for name in STRATA},
        "stratum_counts_selected": {name: len(chosen[name]) for name in STRATA},
        "sequences": [
            {"sequence": name, "stratum": next(s for s in STRATA if name in chosen[s]),
             "attributes": attributes_of[name]}
            for name in selected
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"selected {len(selected)}/{total} sequences -> {out_dir / 'sequences.txt'}")
    for name in STRATA:
        print(f"  {name:<18} {len(chosen[name]):>3} / {len(by_stratum[name]):>3} available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
