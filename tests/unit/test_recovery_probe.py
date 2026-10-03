from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from recovery_probe import (InfraredIndex, apply_corruption, build_mask,  # noqa: E402
                            constant_step, find_infrared, frame_number,
                            target_region_mask)
from codetrack.data.corruption.corruption import corrupt_tokens  # noqa: E402


def test_feature_noise_only_changes_the_selected_random_tokens():
    rng = np.random.default_rng(4)
    tokens = torch.randn(16, 32)
    mask = build_mask("feat_noise", 16, 4, 0.25, rng)

    noisy = apply_corruption("feat_noise", tokens, mask, 0.4, rng)

    assert int(mask.sum()) == 4
    assert torch.equal(noisy[~mask], tokens[~mask])
    assert not torch.equal(noisy[mask], tokens[mask])


def test_block_replacement_matches_each_replaced_token_norm():
    rng = np.random.default_rng(8)
    tokens = torch.randn(16, 32)
    mask = build_mask("block_replace", 16, 4, 0.25, rng)

    replaced = apply_corruption("block_replace", tokens, mask, 0.4, rng)

    assert torch.allclose(replaced[mask].norm(dim=-1), tokens[mask].norm(dim=-1),
                          atol=1e-6, rtol=1e-6)
    assert torch.equal(replaced[~mask], tokens[~mask])


def test_target_region_uses_patch_centers_and_has_a_nearest_cell_fallback():
    centered = target_region_mask(np.array([0.25, 0.25, 0.5, 0.5]), grid=4)
    tiny = target_region_mask(np.array([0.48, 0.48, 0.01, 0.01]), grid=4)

    assert int(centered.sum()) == 4
    assert int(tiny.sum()) == 1


def test_training_block_erasure_uses_exact_ratio_on_a_two_dimensional_grid():
    tokens = torch.randn(2, 256, 32)
    corrupted, mask = corrupt_tokens(
        tokens, "tok_block_erase", 0.2, generator=torch.Generator().manual_seed(11))

    assert torch.equal(mask.sum(dim=1), torch.full((2,), 51.0))
    assert torch.equal(corrupted[mask.bool()], torch.zeros_like(corrupted[mask.bool()]))
    assert torch.equal(corrupted[~mask.bool()], tokens[~mask.bool()])


# ------------------------------------------------------------------ frame pairing
def _make_sequence(tmp_path, visible_names, infrared_names):
    visible = tmp_path / "visible"
    infrared = tmp_path / "infrared"
    visible.mkdir(parents=True, exist_ok=True)
    infrared.mkdir(parents=True, exist_ok=True)
    for name in visible_names:
        (visible / name).touch()
    for name in infrared_names:
        (infrared / name).touch()
    return sorted(visible.glob("*.jpg")), infrared


def test_pairing_handles_identical_names_and_modal_prefix_swap(tmp_path):
    # form 1: 000158.jpg <-> infrared/000158.jpg
    same_name_frames, same_name_dir = _make_sequence(
        tmp_path / "identical", ["000158.jpg", "000159.jpg"],
        ["000158.jpg", "000159.jpg", "i00159.jpg"])
    # form 2: v000158.jpg <-> i000158.jpg
    swap_frames, swap_dir = _make_sequence(
        tmp_path / "swapped", ["v000158.jpg", "v000159.jpg"],
        ["i000158.jpg", "i000159.jpg"])

    assert find_infrared(same_name_frames[0], same_name_dir) == same_name_dir / "000158.jpg"
    assert find_infrared(same_name_frames[1], same_name_dir) == same_name_dir / "000159.jpg"
    assert find_infrared(swap_frames[0], swap_dir) == swap_dir / "i000158.jpg"
    assert find_infrared(swap_frames[1], swap_dir) == swap_dir / "i000159.jpg"

    index = InfraredIndex(swap_frames, swap_dir)
    assert index.match_counts["name"] == 0
    for frame in swap_frames:
        assert index.find(frame) == swap_dir / ("i" + frame.name[1:])
    assert index.match_counts["prefix"] == 2


def test_pairing_matches_modality_letter_suffix_and_number_width(tmp_path):
    # form 3: 000001v.jpg <-> 000001.jpg, form 4: unrelated prefix, same frame number
    suffix_frames, suffix_dir = _make_sequence(
        tmp_path / "suffix", ["000001v.jpg", "000002v.jpg"],
        ["000001.jpg", "000002.jpg"])
    number_frames, number_dir = _make_sequence(
        tmp_path / "number", ["ab_000158.jpg"], ["i000158.jpg"])

    assert find_infrared(suffix_frames[0], suffix_dir) == suffix_dir / "000001.jpg"
    assert find_infrared(number_frames[0], number_dir) == number_dir / "i000158.jpg"
    index = InfraredIndex(number_frames, number_dir)
    assert index.find(number_frames[0]) == number_dir / "i000158.jpg"
    assert index.match_counts["number"] == 1


def test_pairing_refuses_an_ambiguous_frame_number(tmp_path):
    # Two TIR files claim frame 0001 and no name-level rule can separate them: refuse
    # rather than flip a coin, and report the frame as unpaired.
    frames, infrared = _make_sequence(
        tmp_path / "ambiguous", ["frame0001.jpg"], ["a0001.jpg", "b0001.jpg"])
    index = InfraredIndex(frames, infrared)

    assert index.find(frames[0]) is None
    assert index.match_counts["unpaired"] == 1

    # The same frame is still resolvable when an unambiguous name-level rule exists.
    swapped_frames, swapped_dir = _make_sequence(
        tmp_path / "swap_wins", ["v0001.jpg"], ["i0001.jpg", "other0001.jpg"])
    assert find_infrared(swapped_frames[0], swapped_dir) == swapped_dir / "i0001.jpg"


def test_pairing_falls_back_to_position_only_for_consistent_number_series(tmp_path):
    # LasHeR's bowblkboy1-quezhen: v0001..v0004 vs i1000..i1003 (constant index offset).
    offset_frames, offset_dir = _make_sequence(
        tmp_path / "offset", [f"v{i:04d}.jpg" for i in range(1, 5)],
        [f"i{i:04d}.jpg" for i in range(1000, 1004)])

    offset_index = InfraredIndex(offset_frames, offset_dir)
    for position, frame in enumerate(offset_frames):
        assert offset_index.find(frame) == offset_dir / f"i{1000 + position:04d}.jpg"
    assert offset_index.match_counts["position"] == 4

    # Inconsistent numbering of the same length must not be paired positionally.
    broken_frames, broken_dir = _make_sequence(
        tmp_path / "broken", [f"v{i:04d}.jpg" for i in range(1, 5)],
        ["i0000.jpg", "i0009.jpg", "i0011.jpg", "i0100.jpg"])
    broken_index = InfraredIndex(broken_frames, broken_dir)
    assert broken_index.positional is None
    assert broken_index.find(broken_frames[0]) is None


def test_pairing_uses_positions_when_the_infrared_names_carry_no_number(tmp_path):
    frames, infrared = _make_sequence(
        tmp_path / "unnumbered", ["v000.jpg", "v001.jpg", "v002.jpg"],
        ["left.jpg", "middle.jpg", "right.jpg"])
    index = InfraredIndex(frames, infrared)

    assert index.positional == sorted(infrared.glob("*.jpg"))
    assert index.find(frames[1]) == infrared / "middle.jpg"
    assert index.match_counts["position"] == 1
    assert index.strategy == "position"


def test_pairing_never_mixes_name_and_position_within_a_sequence(tmp_path):
    # A mixed rule would pair the overlapping numbering range by name (v1000 -> i1000)
    # while leaving v0001/v0002 unmatched -- two contradictory readings of one sequence.
    frames, infrared = _make_sequence(
        tmp_path / "mixed", ["v0001.jpg", "v0002.jpg", "v1000.jpg", "v1001.jpg"],
        ["i1000.jpg", "i1001.jpg", "i1002.jpg"])
    index = InfraredIndex(frames, infrared)

    assert index.strategy == "refused"
    assert index.notes
    assert all(index.find(frame) is None for frame in frames)
    assert index.match_counts["unpaired"] == len(frames)


def test_pairing_prefers_names_for_the_whole_sequence_when_every_frame_matches(tmp_path):
    frames, infrared = _make_sequence(
        tmp_path / "all_named", ["v000.jpg", "v001.jpg", "v002.jpg"],
        ["i000.jpg", "i001.jpg", "i002.jpg"])
    index = InfraredIndex(frames, infrared)

    assert index.strategy == "name"
    assert index.find(frames[0]) == infrared / "i000.jpg"
    assert index.match_counts["prefix"] == 1


def test_frame_number_and_constant_step_use_the_longest_digit_run():
    assert frame_number("192.168.1.65_02_2019120715172198000") == 2019120715172198000
    assert frame_number("v000158") == 158
    assert frame_number("nomatch") is None
    assert constant_step([0, 1, 2, 3]) == 1
    assert constant_step([1000, 1001, 1002]) == 1
    assert constant_step([0, 2, 4]) == 2
    assert constant_step([0, 1, 3]) is None
    assert constant_step([7]) == 0
