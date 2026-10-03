#!/usr/bin/env bash
# Thin wrapper around `codetrack-demo`.
set -euo pipefail

python -m codetrack.cli.demo "$@"
