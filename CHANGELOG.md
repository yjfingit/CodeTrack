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

## [0.1.0] - 2026-10-03

### Added
- Repository initialized.
