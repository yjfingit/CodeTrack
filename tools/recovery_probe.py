#!/usr/bin/env python
"""Measure decoder messages and output LayerNorm on target-aligned LasHeR crops.

The probe uses the same ground-truth box to make template and search crops, corrupts RGB
only by default, and reports errors separately for damaged RGB tokens, healthy RGB tokens,
the target region, and untouched TIR.  The four output cells are evaluated from one model
forward and the same corruption sample:

* ``v``: corrupted identity baseline;
* ``ln_v``: output LayerNorm applied without decoder updates;
* ``v_plus_d``: the gated residual update before output LayerNorm;
* ``ln_v_plus_d``: the complete current decoder output.

RGB/TIR pairing is resolved by :class:`InfraredIndex`, because LasHeR mixes several file
naming conventions (``v000158.jpg``/``i000158.jpg``, ``000158.jpg``/``000158.jpg``,
``000001v.jpg``/``000001.jpg``, and constant index offsets).  The report records how many
frames each rule paired and lists every skipped frame, so a wrong pairing cannot hide.

Usage::

    python tools/recovery_probe.py --checkpoint outputs/<exp>/final.pth --sequences 20
    python tools/recovery_probe.py --checkpoint ... --regimes block_replace random
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import cv2
import numpy as np
import torch

from codetrack.data.transforms.sample import to_tensor  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def frame_number(stem: str) -> Optional[int]:
    """Integer value of the longest digit run in a file stem, or ``None``."""
    runs = re.findall(r"\d+", stem)
    return int(max(runs, key=len)) if runs else None


def constant_step(numbers: Sequence[int]) -> Optional[int]:
    """The common difference of an arithmetic sequence, else ``None``."""
    if len(numbers) < 2:
        return 0
    steps = {second - first for first, second in zip(numbers, numbers[1:])}
    return next(iter(steps)) if len(steps) == 1 else None


class InfraredIndex:
    """Resolve the TIR frame paired with each RGB frame of one LasHeR sequence.

    LasHeR is not internally consistent, and the old ``"i" + name[1:]`` rule silently
    dropped every frame of the sequences it did not fit.  Five on-disk conventions occur
    (counts from the shipped ``testingset`` / ``trainingset`` listings):

    ====================  ====================  ==================================
    visible               infrared              note
    ====================  ====================  ==================================
    ``v000158.jpg``       ``i000158.jpg``       prefix swap, the common case
    ``000158.jpg``        ``000158.jpg``        identical names
    ``000001v.jpg``       ``000001.jpg``        modality letter as a suffix
    ``v00.jpg``           ``i00.jpg``           same number, different width
    ``v0001.jpg``         ``i1000.jpg``         constant index offset (only pairable
                                                positionally), e.g. LasHeR's
                                                ``bowblkboy1-quezhen``
    ====================  ====================  ==================================

    The pairing *strategy is chosen once per sequence*, never frame by frame.  Mixing rules
    is how an offset sequence such as ``bowblkboy1-quezhen`` (visible ``v0001..v1264``,
    infrared ``i1000..``) silently produced contradictory pairs: the overlapping numbering
    range paired by name while the rest stayed unpaired, which cannot both be right.

    Order of preference:

    1. **every** frame resolves by name -- identical name (other image extensions tried),
       modality-prefix swap, the modality letter as a suffix, or identical frame number;
    2. otherwise positional alignment, accepted only when both listings have the same
       length and their frame numbers cannot disagree (both arithmetic with the same
       step) -- this covers ``orange`` (camera-timestamp IR names) and constant index
       offsets;
    3. otherwise the sequence is refused entirely and reported, rather than half-paired.

    Ambiguous frame numbers are refused rather than guessed, and ``match_counts`` records
    which rule actually paired each frame.
    """

    def __init__(self, rgb_frames: Sequence[Path], infrared_dir: Path,
                 allow_positional: bool = True) -> None:
        self.infrared_dir = Path(infrared_dir)
        self.by_name: Dict[str, Path] = {}
        self.by_number: Dict[int, Optional[Path]] = {}
        self.positional: Optional[List[Path]] = None
        self.strategy = "refused"
        self.notes: List[str] = []
        self.match_counts: Dict[str, int] = {"name": 0, "extension": 0, "prefix": 0,
                                             "number": 0, "position": 0, "unpaired": 0}
        self._tiers: Dict[Path, str] = {}
        self._planned: Dict[Path, Path] = {}
        rgb_frames = list(rgb_frames)
        self._positions: Dict[Path, int] = {path: i for i, path in enumerate(rgb_frames)}
        ir_frames = self._list_frames()
        for path in ir_frames:
            self.by_name.setdefault(path.name, path)
            number = frame_number(path.stem)
            if number is None:
                continue
            if number not in self.by_number:
                self.by_number[number] = path
            elif self.by_number[number] != path:
                self.by_number[number] = None       # two files claim this number: no guess
        self._plan(rgb_frames, ir_frames, allow_positional)

    def _list_frames(self) -> List[Path]:
        if not self.infrared_dir.is_dir():
            return []
        frames = [path for path in sorted(self.infrared_dir.iterdir())
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        return frames

    def _plan(self, rgb_frames: List[Path], ir_frames: List[Path],
              allow_positional: bool) -> None:
        named: Dict[Path, Path] = {}
        tier_of: Dict[Path, str] = {}
        missing = 0
        for frame in rgb_frames:
            match = self._find_by_name(frame)
            if match is None:
                missing += 1
                continue
            named[frame] = match[0]
            tier_of[frame] = match[1]
        if named and not missing:
            self.strategy = "name"
            self._planned = named
            self._tiers = tier_of
            return
        positional = self._positional_alignment(rgb_frames, ir_frames, allow_positional)
        if positional is not None:
            self.strategy = "position"
            self.positional = positional
            self._planned = dict(zip(rgb_frames, positional))
            self._tiers = {frame: "position" for frame in rgb_frames}
            return
        if named:
            self.notes.append(
                f"refused: {missing}/{len(rgb_frames)} frames have no name-level pair and "
                "positional alignment is not verifiable")
        self._planned = {}
        self._tiers = {}

    def _positional_alignment(self, rgb_frames: List[Path], ir_frames: List[Path],
                              allow_positional: bool) -> Optional[List[Path]]:
        if not allow_positional:
            return None
        if len(rgb_frames) != len(ir_frames) or not rgb_frames:
            return None
        rgb_numbers = [frame_number(path.stem) for path in rgb_frames]
        ir_numbers = [frame_number(path.stem) for path in ir_frames]
        if all(number is not None for number in rgb_numbers + ir_numbers):
            rgb_step = constant_step([number for number in rgb_numbers if number is not None])
            ir_step = constant_step([number for number in ir_numbers if number is not None])
            if rgb_step is None or ir_step is None or rgb_step != ir_step:
                return None
        return ir_frames

    def _find_by_name(self, rgb_path: Path) -> Optional[tuple]:
        for name, tier in self._candidates_with_tier(rgb_path):
            found = self.by_name.get(name)
            if found is not None:
                return found, tier
        number = frame_number(rgb_path.stem)
        if number is not None:
            found = self.by_number.get(number)
            if found is not None:
                return found, "number"
        return None

    def candidates(self, rgb_path: Path) -> List[str]:
        """File names to try in ``infrared/`` for one RGB frame, most specific first."""
        return [name for name, _ in self._candidates_with_tier(rgb_path)]

    @staticmethod
    def _candidates_with_tier(rgb_path: Path) -> List[tuple]:
        stem, suffix = rgb_path.stem, rgb_path.suffix
        named = [(rgb_path.name, "name")]
        for other_suffix in IMAGE_SUFFIXES:
            if other_suffix != suffix.lower():
                named.append((f"{stem}{other_suffix}", "extension"))
        letters = [character for character in stem if character.isalpha()]
        digits_only = "".join(character for character in stem if not character.isalpha())
        if letters:
            first, rest = stem[0], stem[1:]
            swapped = ("i" if first == "v" else "v") + rest
            named.extend([(f"{swapped}{suffix}", "prefix"), (swapped, "prefix")])
        if digits_only and digits_only != stem:
            named.extend([(f"{digits_only}{suffix}", "prefix"), (digits_only, "prefix")])
        return named

    def find(self, rgb_path: Path) -> Optional[Path]:
        """The TIR frame paired with ``rgb_path``, or ``None`` when there is no safe pair."""
        rgb_path = Path(rgb_path)
        found = self._planned.get(rgb_path)
        if found is None:
            self.match_counts["unpaired"] += 1
            return None
        self.match_counts[self._tiers[rgb_path]] += 1
        return found


def find_infrared(rgb_path: Path, infrared_dir: Path) -> Optional[Path]:
    """Pair one RGB frame by name or frame number.

    Positional alignment needs the whole listing to be validated (equal length, consistent
    numbering), so it is deliberately unavailable here; use :class:`InfraredIndex` with the
    sequence's full RGB list when the index-offset convention must be supported.
    """
    return InfraredIndex([Path(rgb_path)], Path(infrared_dir),
                         allow_positional=False).find(Path(rgb_path))


def build_mask(kind: str, n: int, grid: int, ratio: float,
               rng: np.random.Generator) -> torch.Tensor:
    """Build an N-token mask in row-major patch order."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    k = max(1, int(round(ratio * n)))
    mask = torch.zeros(n, dtype=torch.bool)
    if kind in ("block", "block_replace"):
        aspect = float(rng.uniform(0.5, 2.0))
        height = max(1, min(grid, int(round(np.sqrt(k / aspect)))))
        width = max(1, min(grid, int(np.ceil(k / height))))
        if height * width < k:
            height = min(grid, int(np.ceil(k / width)))
        r0 = int(rng.integers(0, grid - height + 1))
        c0 = int(rng.integers(0, grid - width + 1))
        positions = np.array([(r0 + dr) * grid + c0 + dc
                              for dr in range(height) for dc in range(width)])
        selected = rng.permutation(positions)[:k]
        mask[torch.from_numpy(selected)] = True
    elif kind in ("random", "feat_noise"):
        mask[torch.from_numpy(rng.permutation(n)[:k])] = True
    else:
        raise ValueError(f"unknown recovery regime {kind!r}")
    return mask


def apply_corruption(kind: str, tokens: torch.Tensor, mask: torch.Tensor,
                     severity: float, rng: np.random.Generator) -> torch.Tensor:
    """Corrupt selected positions of an ``N x C`` feature tensor."""
    if not bool(mask.any()):
        return tokens
    mask = mask.to(device=tokens.device, dtype=torch.bool)
    out = tokens.clone()
    if kind in ("random", "block"):
        out[mask] = 0.0
    elif kind == "block_replace":
        # Replace with healthy donor features, then match each replaced token's norm.  The
        # old neighborhood-mean fill changed the feature-energy distribution substantially.
        healthy = tokens[~mask]
        if healthy.numel() == 0:
            healthy = tokens
        donor_order = torch.randperm(healthy.shape[0], device=tokens.device)
        donor = healthy[donor_order[torch.arange(int(mask.sum()), device=tokens.device)
                                    % healthy.shape[0]]]
        target_norm = tokens[mask].norm(dim=-1, keepdim=True)
        donor_norm = donor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out[mask] = donor * (target_norm / donor_norm)
    elif kind == "feat_noise":
        seed = int(rng.integers(0, 2**31 - 1))
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        noise = torch.randn(tokens[mask].shape, generator=generator,
                            device=tokens.device, dtype=tokens.dtype)
        out[mask] += noise * (float(severity) * tokens.std().clamp(min=1e-6))
    else:
        raise ValueError(f"unknown recovery regime {kind!r}")
    return out


def target_region_mask(box_xywh: np.ndarray, grid: int) -> torch.Tensor:
    """Mark patch centers inside a ground-truth box in normalized search coordinates."""
    x, y, width, height = [float(v) for v in box_xywh]
    cx, cy = x + width / 2.0, y + height / 2.0
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    selected = ((xx >= x) & (xx <= x + width) & (yy >= y) & (yy <= y + height))
    if not bool(selected.any()):
        nearest = torch.argmin((xx - cx).square() + (yy - cy).square())
        selected.view(-1)[nearest] = True
    return selected.flatten()


def token_error(pred: torch.Tensor, clean: torch.Tensor, region: torch.Tensor) -> float:
    errors = (pred.float() - clean.float()).abs().mean(dim=-1)
    region = region.to(device=errors.device, dtype=torch.bool)
    if not bool(region.any()):
        return float("nan")
    return float(errors[region].mean())


def read_annotations(path: Path) -> np.ndarray:
    return np.array([[float(value) for value in line.replace("\t", ",").split(",")[:4]]
                     for line in path.read_text().splitlines() if line.strip()],
                    dtype=np.float32)


def crop_box(trainer, image: np.ndarray, box: np.ndarray, factor: float,
             output_size: int) -> np.ndarray:
    x, y, width, height = [float(v) for v in box]
    cx, cy = x + width / 2.0, y + height / 2.0
    side = float(np.sqrt(max(width, 1.0) * max(height, 1.0))) * factor
    return trainer._crop_resize(image, cx, cy, side, output_size)


def list_sequences(root: Path, subset: str, limit: int) -> List[str]:
    list_name = ("testingsetList.txt" if subset.startswith("test")
                 else "trainingsetList.txt")
    sequences = [line.strip() for line in (root / list_name).read_text().splitlines()
                 if line.strip()]
    return sequences[:limit]


def prepare_plans(root: Path, subset: str, sequences: Sequence[str], frames: int
                  ) -> tuple:
    """Resolve the RGB/TIR frame pairs of every usable sequence, once.

    Pairing is regime independent, so both ``recovery_probe`` and ``gate_probe`` resolve it
    here and report exactly how each frame was paired.  LasHeR's IR numbering is not
    consistent across sequences (see :class:`InfraredIndex`).

    Returns ``(plans, skipped, pairing_counts, requested_frames)``.  Each plan holds the
    sequence name, its visible frame paths, ``(frame_index, tir_path)`` pairs, the
    annotation array and the template's IR frame.
    """
    plans: List[Dict[str, object]] = []
    skipped: List[str] = []
    pairing_counts: Dict[str, int] = {"name": 0, "extension": 0, "prefix": 0, "number": 0,
                                      "position": 0, "unpaired": 0}
    requested_frames = 0
    for sequence in sequences:
        seq_dir = root / subset / sequence
        visible_dir = seq_dir / "visible"
        rgb_frames = sorted(visible_dir.glob("*.jpg")) or sorted(visible_dir.glob("*.png"))
        annotations_path = root / "annos" / f"{sequence}.txt"
        if not annotations_path.exists() or len(rgb_frames) < 2:
            skipped.append(f"{sequence}: missing annotations or fewer than two frames")
            continue
        annotations = read_annotations(annotations_path)
        count = min(len(rgb_frames), len(annotations), 1 + frames)
        if count < 2:
            skipped.append(f"{sequence}: no annotated search frames")
            continue
        index = InfraredIndex(rgb_frames, seq_dir / "infrared")
        first_ir = index.find(rgb_frames[0])
        if first_ir is None:
            skipped.append(f"{sequence}: no infrared for {rgb_frames[0].name}")
            continue
        requested_frames += count - 1
        pairs: List[List[object]] = []
        for frame_index in range(1, count):
            tir_path = index.find(rgb_frames[frame_index])
            if tir_path is None:
                skipped.append(f"{sequence}/{rgb_frames[frame_index].name}: "
                               "no infrared pair")
                continue
            pairs.append([frame_index, tir_path])
        if not pairs:
            skipped.append(f"{sequence}: no pairable search frames")
            continue
        for key, value in index.match_counts.items():
            pairing_counts[key] += value
        plans.append({"sequence": sequence, "frames": rgb_frames, "pairs": pairs,
                      "boxes": annotations, "template_ir": first_ir})
    return plans, skipped, pairing_counts, requested_frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    ap.add_argument("--subset", default="testingset")
    ap.add_argument("--sequences", type=int, default=20)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--severity", type=float, default=0.4)
    ap.add_argument("--regimes", default="block,block_replace,random,feat_noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/recovery_probe.json")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "recprobe")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint,
                        report["epoch"])
    model = trainer.model.eval()

    root = Path(args.root)
    sequences = list_sequences(root, args.subset, args.sequences)

    grid = model.grid
    n_tok = model.num_variables
    tpl_size = model.template_size
    search_size = model.img_size
    data_cfg = cfg.get("data", {})
    template_factor = float(data_cfg.get("template_factor", 2.0))
    search_factor = float(data_cfg.get("search_factor", 4.0))
    cell_names = ("v", "ln_v", "v_plus_d", "ln_v_plus_d")
    categories = ("rgb_damaged", "rgb_healthy", "rgb_target", "tir_all", "tir_target",
                 "both_target")
    rows: List[Dict[str, object]] = []
    plans, skipped, pairing_counts, requested_frames = prepare_plans(
        root, args.subset, sequences, args.frames)

    for regime in [value.strip() for value in args.regimes.split(",") if value.strip()]:
        rng = np.random.default_rng(args.seed)
        measurements = {cell: {category: [] for category in categories}
                        for cell in cell_names}
        for plan in plans:
            sequence = str(plan["sequence"])
            rgb_frames = plan["frames"]
            annotations = plan["boxes"]
            first_ir = plan["template_ir"]
            first_rgb_raw = cv2.imread(str(rgb_frames[0]))
            first_tir_raw = cv2.imread(str(first_ir), cv2.IMREAD_GRAYSCALE)
            if first_rgb_raw is None or first_tir_raw is None:
                skipped.append(f"{sequence}: unreadable template frame")
                continue
            first_rgb = cv2.cvtColor(first_rgb_raw, cv2.COLOR_BGR2RGB)
            template_rgb = crop_box(trainer, first_rgb, annotations[0], template_factor,
                                    tpl_size)
            template_tir = crop_box(trainer, first_tir_raw, annotations[0], template_factor,
                                    tpl_size)
            tpl_rgb_t = to_tensor(template_rgb, 3, (0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
            tpl_tir_t = to_tensor(template_tir, 1, (0.449,), (0.226,)).unsqueeze(0).to(
                trainer.device)

            for frame_index, tir_path in plan["pairs"]:
                rgb_path = rgb_frames[frame_index]
                rgb_raw = cv2.imread(str(rgb_path))
                tir_raw = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
                if rgb_raw is None or tir_raw is None:
                    skipped.append(f"{sequence}/{rgb_path.name}: unreadable frame")
                    continue
                rgb_full = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
                box = annotations[frame_index]
                search_rgb = crop_box(trainer, rgb_full, box, search_factor, search_size)
                search_tir = crop_box(trainer, tir_raw, box, search_factor, search_size)
                search_rgb_t = to_tensor(search_rgb, 3, (0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
                search_tir_t = to_tensor(search_tir, 1, (0.449,), (0.226,)).unsqueeze(0).to(
                    trainer.device)

                mask = build_mask(regime, n_tok, grid, args.ratio, rng).to(trainer.device)
                with torch.no_grad():
                    feats = model.backbone((tpl_rgb_t, search_rgb_t),
                                           (tpl_tir_t, search_tir_t), return_inter=True)
                    clean_rgb = feats["x_r"][0]
                    clean_tir = feats["x_t"][0]
                    noisy_rgb = apply_corruption(regime, clean_rgb, mask, args.severity, rng)
                    out = model._code_path(
                        feats["z_r"], noisy_rgb.unsqueeze(0), feats["z_t"],
                        clean_tir.unsqueeze(0),
                        {"inter_r": feats["inter_r"], "inter_t": feats["inter_t"]})

                search_side = float(np.sqrt(max(float(box[2]), 1.0) *
                                            max(float(box[3]), 1.0))) * search_factor
                box_w = float(box[2]) / search_side
                box_h = float(box[3]) / search_side
                target_mask = target_region_mask(
                    np.array([0.5 - box_w / 2.0, 0.5 - box_h / 2.0, box_w, box_h],
                             dtype=np.float32), grid).to(trainer.device)
                views = {
                    "v": (noisy_rgb, clean_tir),
                    "ln_v": (out["norm_only_rgb"][0], out["norm_only_tir"][0]),
                    "v_plus_d": (out["pre_norm_rgb"][0], out["pre_norm_tir"][0]),
                    "ln_v_plus_d": (out["corrected_rgb"][0], out["corrected_tir"][0]),
                }
                rgb_healthy = ~mask
                for cell, (pred_rgb, pred_tir) in views.items():
                    rgb_damaged = token_error(pred_rgb, clean_rgb, mask)
                    rgb_healthy_error = token_error(pred_rgb, clean_rgb, rgb_healthy)
                    rgb_target_error = token_error(pred_rgb, clean_rgb, target_mask)
                    tir_all_error = token_error(pred_tir, clean_tir,
                                                torch.ones_like(mask, dtype=torch.bool))
                    tir_target_error = token_error(pred_tir, clean_tir, target_mask)
                    both_target_error = 0.5 * (rgb_target_error + tir_target_error)
                    values = (rgb_damaged, rgb_healthy_error, rgb_target_error,
                              tir_all_error, tir_target_error, both_target_error)
                    for category, value in zip(categories, values):
                        measurements[cell][category].append(value)

        if not measurements["v"]["rgb_damaged"]:
            print(f"[{regime}] no usable frames")
            continue
        errors = {
            cell: {category: float(np.nanmean(values)) if values else float("nan")
                   for category, values in by_category.items()}
            for cell, by_category in measurements.items()
        }
        damaged = {cell: errors[cell]["rgb_damaged"] for cell in cell_names}
        row = {
            "regime": regime,
            "ratio": args.ratio,
            "severity": args.severity,
            "frames": len(measurements["v"]["rgb_damaged"]),
            "errors": errors,
            "damaged_rgb_gain_pre_norm": 1.0 - damaged["v_plus_d"] /
            max(damaged["v"], 1e-8),
            "damaged_rgb_gain_post_norm": 1.0 - damaged["ln_v_plus_d"] /
            max(damaged["v"], 1e-8),
            "layernorm_effect_without_messages": errors["ln_v"]["rgb_damaged"] -
            errors["v"]["rgb_damaged"],
            "layernorm_effect_with_messages": errors["ln_v_plus_d"]["rgb_damaged"] -
            errors["v_plus_d"]["rgb_damaged"],
            "rgb_healthy_damage_post_norm": errors["ln_v_plus_d"]["rgb_healthy"] -
            errors["v"]["rgb_healthy"],
            "tir_damage_post_norm": errors["ln_v_plus_d"]["tir_all"] -
            errors["v"]["tir_all"],
        }
        rows.append(row)
        print(f"[{regime}] frames={row['frames']} "
              f"damaged RGB: v={damaged['v']:.4f}, LN(v)={damaged['ln_v']:.4f}, "
              f"v+D={damaged['v_plus_d']:.4f}, LN(v+D)={damaged['ln_v_plus_d']:.4f}; "
              f"target={errors['ln_v_plus_d']['both_target']:.4f}, "
              f"healthy RGB damage={row['rgb_healthy_damage_post_norm']:.4f}, "
              f"TIR damage={row['tir_damage_post_norm']:.4f}")

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "subset": args.subset,
        "requested_sequences": args.sequences,
        "usable_sequences": len(plans),
        "frames_per_sequence_cap": args.frames,
        "requested_search_frames": requested_frames,
        "ratio": args.ratio,
        "severity": args.severity,
        "seed": args.seed,
        "regimes": [value.strip() for value in args.regimes.split(",") if value.strip()],
        "pairing": pairing_counts,
        "corruption_target": "rgb",
        "measured_rows": rows,
        "skipped": skipped,
    }
    print("\nFour cells are measured from the same RGB-only corruption sample; the TIR "
          "branch is left uncorrupted and is reported separately.  Positive gains mean "
          "lower feature error.")
    print(f"paired {requested_frames} search frames over {len(plans)} sequences "
          f"({pairing_counts}); skipped {len(skipped)} items.")
    if skipped:
        print(f"First skipped entries: {skipped[:3]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
