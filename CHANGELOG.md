# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial repository skeleton: package layout, configs, docs, scripts, tests.
- Architecture documentation with tensor shapes (`docs/architecture.md`).
- `tools/git-hooks/post-commit` — background auto-push to `origin`, installed by
  `tools/install-git-hooks.sh`; `tools/sync-to-github.sh` as a manual fallback.
- `tools/install-gh.sh` — persistent GitHub CLI install (kept on the data disk) with
  device-code login and `gh auth setup-git` wiring.
- **Full CodeTrack implementation following the architecture figure**
  (`codetrack/models/`): shared OSTrack ViT-B/16 backbone with lossless pretrained
  loading, target codebook encoder, reliability estimator, target candidate selector,
  reliability-adaptive Tanner graph, visual syndrome, error locator, neural
  belief-propagation decoder, FPN fusion and the centre/corner tracking head.
- Data and training stack (`codetrack/data/`, `codetrack/engine/`): LasHeR reader,
  template/search sampling, channel corruption simulator, the four-term objective
  `L_track + 1.0·L_detect + 2.0·L_correct + 0.1·L_identity` with a clean teacher, and a
  trainer with a staged codec warm-up.
- Evaluation with PR / SR / NPR (`codetrack/metrics/`, `codetrack/cli/eval.py`).
- `tools/check_shapes.py` — audits all 21 architecture-figure tensor shapes in one
  forward pass (currently 21/21 OK).
- Experiment configs: `lasher_vitb`, `lasher_vitb_corrupt`, `lasher_vitb_minimal`.
- `codetrack/utils/runtime.py` — per-process CPU thread caps (`OMP`/`MKL`/OpenBLAS/
  PyTorch/OpenCV) shared by training, evaluation and every probe, so parallel runs stay
  measurable.
- `tools/pairing_audit.py` — read-only RGB/TIR pairing audit over both LasHeR subsets.
- `tools/gate_probe.py` — learned / zero / full / oracle severity-gate policies on one
  corruption sample, with the applied-vs-ideal update cosine and norm ratio.
- `tools/select_validation_sequences.py` — seeded, stratified validation split from
  LasHeR's own challenge attributes (`AttriSeqsTxt/`).
- `tools/validate_sequences.sh` — one frozen checkpoint x one condition matrix on a fixed
  sequence list, several conditions in parallel, with a `run_manifest.json` per run.
- `tools/summarize_paired_conditions.py` — sequence-level paired statistics (bootstrap
  interval, paired t, Wilcoxon, Holm-corrected p) and cross-checkpoint comparisons, with a
  secondary view restricted to the sequences the reference arm actually tracks, a per-stratum
  table, and `--screen-run` to evaluate the pre-registered strength screen
  (20-40 % relative drop, 60-80 % retained SR, interval excludes zero) per condition.
- `tools/horizon_probe.py` — per-sequence IoU trace, first frame after which the rolling mean
  IoU never recovers above 0.5, and the loss-time distribution, because an SR average cannot
  distinguish slow drift from a tracker that was never on the target.
- `codetrack/cli/eval.py --sequence-list` and `Trainer.evaluate(sequence_list=...)` —
  evaluate an explicit sequence list; the default `None` keeps the historical
  "first N of the subset" path bit-identical.
- `model.mlp_hidden` (convolution-free config key, default `0`) — names the MLP baseline's
  hidden width instead of always solving it from the BP parameter count. `0` keeps the
  auto/parameter-matched width, so the baseline path is unchanged; an already-trained MLP arm
  (trained at 2048 hidden with the earlier formula, 6.307 M decoder parameters) can now be
  rebuilt and loaded with `model.mlp_hidden=2048`.
- `tools/validate_sequences.sh`: `token`-level conditions `tok_block_tir_04` and
  `tok_block_both_04`, completing the ratio 0.2 / 0.4 curve for RGB-only, TIR-only and
  dual-modality corruption.

### Fixed
- `codetrack/utils/checkpoint.py`: `load_checkpoint` now refuses to load a checkpoint whose
  stored `model.decoder_mode` differs from the model being built. An `off` arm loaded into a
  `bp` model left 16 message-passing tensors at their random initialisation (and the MLP arm
  14 + 8 unexpected) while `strict=False` reported it only in a list nobody read, so an
  evaluation could silently measure a randomly wired decoder. `Trainer.evaluate` also logs a
  warning whenever a load reports missing or unexpected keys.
- `tools/validate_sequences.sh`: takes `OVERRIDES` for the model config a checkpoint requires
  and records it in `run_manifest.json`.
- `tools/recovery_probe.py`: RGB/TIR pairing no longer assumes `infrared/i<rest>`. The
  resolver decides **one strategy per sequence** (identical name, modality-prefix swap,
  modality letter as a suffix, identical frame number, or positional alignment that is
  only accepted when both listings are provably index-aligned) and refuses a sequence
  instead of mixing rules -- which had paired an offset sequence half by name and half not
  at all. On the standard 20-sequence selection the old rule silently dropped 72/240
  frames; over `testingset` it left 16 257/220 703 frames unpaired, and it skipped any
  sequence whose visible frames are not `v<digits>.jpg`. Probes now report the pairing
  counts, the usable frame count and every skipped item.
- `codetrack/engine/trainer.py`: image-level corruption conditions crashed with
  `KeyError: 'corrupted_rgb'` in `_collect_diagnostics`, and `evaluate` swallowed it as
  "skip <sequence>" -- every RGB low-light, RGB occlusion and TIR crossover run scored
  zero sequences. The clean-reference pass also reused the *degraded* frame, which made
  the syndrome AUROC of those conditions 0.5 by construction; it now uses the pristine
  frame.
- `tools/diagnostics.py`: `diagnostics.json` stores the invocation it was produced with.
  The same checkpoint scores 66.60 SR at `--frames 10` and 50.90 at `--frames 30`, and the
  artifact previously could not say which setting it came from.

## [0.1.0] - 2026-10-03

### Added
- Repository initialized.
