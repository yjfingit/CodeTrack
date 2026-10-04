#!/usr/bin/env python
"""Which code produced each number in ``docs/results.md`` 6.0?

The 073c223 round made this necessary.  The commit changed the decoder loop's *state semantics*
for every non-identity output mode while its message said the default path was untouched, so arms
trained before it were then evaluated by a different function -- and nothing in the artifacts made
that visible.  Section 6.10 read the resulting mismatch between arm A and the shipped checkpoint
as "the sampling RNG stream shifted".

This tool walks ``outputs/*/`` and reports, per run directory, the training provenance recorded at
train time and the evaluation provenance recorded at eval time, then flags the pairs that cannot
be compared:

* ``eval_before_train`` -- the evaluation ran on code older than the checkpoint it loads;
* ``dirty_at_eval`` / ``dirty_at_train`` -- the tree had uncommitted changes, so the commit alone
  does not reproduce the number;
* ``missing_provenance`` -- the run predates ``run_provenance.json`` / the manifest rewrite, which
  is itself the finding: those numbers cannot be attributed and must be recomputed;
* ``state_mode_changed`` -- the training and evaluation disagree about
  ``decoder_state_mode``/``decoder_output``/``gate_always_one``, i.e. the evaluated network is
  not the trained one.

Usage::

    python tools/run_provenance_audit.py --out outputs/run_provenance_audit.json

The output name deliberately does **not** end in ``run_provenance.json``: the tool scans for that
name, and writing its own report under it would make the report read itself as a training run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.provenance import git_provenance  # noqa: E402

#: Keys whose disagreement means "the evaluated network is not the trained network".
FUNCTION_KEYS = ("decoder_mode", "decoder_output", "decoder_state_mode", "gate_always_one")


def commit_order(repo: Path, commit: str) -> Optional[int]:
    """Position of ``commit`` on the first-parent history, or ``None`` if it is unknown here."""
    try:
        result = subprocess.run(["git", "rev-list", "--first-parent", "HEAD"], cwd=str(repo),
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    history = result.stdout.split()
    return history.index(commit) if commit in history else None


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect(outputs: Path, repo: Path) -> List[Dict[str, Any]]:
    """One row per training run or evaluation manifest found anywhere under ``outputs``.

    Both file names are searched recursively because the two families are laid out differently:
    a training run is ``outputs/p2_A/{run_provenance.json,resolved_config.json,final.pth}`` while
    an evaluation campaign is ``outputs/validation_v1/<label>/<condition>/metrics.json`` with the
    manifest at the campaign root.  A one-level listing silently missed every evaluation.
    """
    rows: List[Dict[str, Any]] = []
    seen: set = set()

    def add(run_dir: Path, kind: str) -> None:
        if (run_dir, kind) in seen:
            return
        seen.add((run_dir, kind))
        train = read_json(run_dir / "run_provenance.json")
        manifest = read_json(run_dir / "run_manifest.json")
        if kind == "eval" and not manifest:
            return
        # A "train" row without provenance is kept on purpose: a checkpoint that cannot be tied
        # to a commit is a finding, not a row to drop.
        if kind == "train" and not train and not (run_dir / "resolved_config.json").exists():
            return
        row: Dict[str, Any] = {
            "run": str(run_dir.relative_to(outputs)),
            "kind": kind,
            "has_train_provenance": bool(train),
            "has_eval_manifest": bool(manifest),
        }
        train_commit = train.get("git_commit")
        eval_prov = manifest.get("eval_provenance") or {}
        eval_commit = eval_prov.get("git_commit")
        nested_train = manifest.get("train_provenance") or {}
        if not train and nested_train:
            train = nested_train
            train_commit = train.get("git_commit")
        row["train_commit"] = train_commit
        row["eval_commit"] = eval_commit
        row["train_dirty"] = train.get("git_dirty")
        row["eval_dirty"] = eval_prov.get("git_dirty")
        row["decoder_state_mode"] = train.get("decoder_state_mode")

        flags: List[str] = []
        if kind == "eval":
            if not train:
                flags.append("missing_train_provenance")
            if not eval_commit:
                flags.append("missing_eval_provenance")
        elif not train:
            flags.append("missing_train_provenance")
        elif not train_commit or train_commit == "unknown":
            flags.append("unknown_train_commit")
        if train.get("git_dirty") is True or eval_prov.get("git_dirty") is True:
            flags.append("dirty_tree")
        if train_commit and eval_commit and train_commit != eval_commit:
            train_pos = commit_order(repo, str(train_commit))
            eval_pos = commit_order(repo, str(eval_commit))
            if train_pos is not None and eval_pos is not None and eval_pos > train_pos:
                flags.append("eval_runs_older_code_than_checkpoint")
            else:
                flags.append("train_eval_commit_mismatch")

        # Do the recorded function switches agree between the checkpoint and the evaluation?
        eval_overrides = " ".join(manifest.get("config_overrides") or [])
        for key in FUNCTION_KEYS:
            trained = train.get(key)
            if trained is None:
                continue
            if key in ("decoder_output", "gate_always_one", "decoder_state_mode"):
                # A trained value that differs from the *default* must appear as an override for
                # the evaluation to rebuild the same function.
                default = {"decoder_output": "post_norm", "gate_always_one": "False",
                           "decoder_state_mode": "recurrent"}.get(key)
                if str(trained) != default and f"{key}=" not in eval_overrides:
                    flags.append(f"eval_missing_override:{key}")
        row["flags"] = flags
        row["comparable"] = not flags
        rows.append(row)

    for path in sorted(outputs.rglob("run_provenance.json")):
        add(path.parent, "train")
    # A checkpoint directory without provenance is the case the audit exists to expose, so it is
    # included rather than skipped: silence here would read as "nothing to report".
    for path in sorted(outputs.rglob("resolved_config.json")):
        if (path.parent / "final.pth").exists() or (path.parent / "last.pth").exists():
            add(path.parent, "train")
    for path in sorted(outputs.rglob("run_manifest.json")):
        add(path.parent, "eval")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="outputs/run_provenance_audit.json")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    repo = Path(args.repo)
    rows = collect(outputs, repo)
    judged = [row for row in rows if row["has_train_provenance"]]
    report = {
        "note": ("Numbers from a run that is not 'comparable' cannot be attributed to a commit "
                 "and must be recomputed before they are used in a comparison."),
        "repo_head": git_provenance(repo),
        "n_runs": len(rows),
        "n_with_train_provenance": len(judged),
        "n_not_comparable": sum(1 for row in rows if not row["comparable"]),
        "runs": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    if not rows:
        print("no run directories carry provenance yet (the schema was just introduced);")
        print("existing runs are therefore unattributable by construction and must be recomputed.")
        return 0
    width = max(len(row["run"]) for row in rows)
    print(f"{'run':<{width}}  {'kind':<5} {'train':<9} {'eval':<9}  flags")
    for row in rows:
        train = (row["train_commit"] or "none")[:7]
        ev = (row["eval_commit"] or "none")[:7]
        print(f"{row['run']:<{width}}  {row['kind']:<5} {train:<9} {ev:<9}  "
              f"{','.join(row['flags']) or 'ok'}")
    print(f"\n{report['n_runs']} runs, {report['n_not_comparable']} not comparable -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
