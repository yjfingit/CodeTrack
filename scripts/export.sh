#!/usr/bin/env bash
# Export a trained checkpoint to a deployment-friendly format.
set -euo pipefail

CKPT="${1:?usage: bash scripts/export.sh <checkpoint> [torchscript|onnx]}"
FORMAT="${2:-torchscript}"

python -m codetrack.cli.export --checkpoint "${CKPT}" --format "${FORMAT}"
