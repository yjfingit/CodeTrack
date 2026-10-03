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

## [0.1.0] - 2026-10-03

### Added
- Repository initialized.
