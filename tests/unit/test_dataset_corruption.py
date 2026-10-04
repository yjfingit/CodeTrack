"""The training pipeline corrupts the *search* frame, not the template.

This was invisible until it mattered: `Trainer.train()` never calls `apply_image_corruption`, so a
reading of that file alone suggests image-level degradation is configured but unused.  It is in
fact applied inside the dataset, and the distinction (search corrupted, template pristine) is part
of the recipe, so it is pinned here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codetrack.data.datasets import build_dataset  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402

DATA_ROOT = Path("/root/autodl-tmp/lab/dataset/LasHeR")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="LasHeR dataset not mounted")
def test_training_dataset_corrupts_the_search_frame_and_keeps_the_template_clean():
    cfg = load_config(str(ROOT / "configs/experiment/lasher_vitb_corrupt.yaml"))
    cfg["data"]["samples_per_epoch"] = 8
    corrupt = build_dataset(cfg)
    clean = build_dataset({**cfg, "corruption": {**cfg["corruption"], "enabled": False}})

    assert corrupt.corruption.enabled
    with_corruption, without = corrupt[0], clean[0]

    # the search frame carries the image-level degradation of the config
    assert not torch.equal(with_corruption["search_rgb"], without["search_rgb"])
    assert not torch.equal(with_corruption["search_tir"], without["search_tir"])
    # the template is always sampled from the pristine first frame
    assert torch.equal(with_corruption["template_rgb"], without["template_rgb"])
    assert torch.equal(with_corruption["template_tir"], without["template_tir"])
