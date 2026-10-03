#!/usr/bin/env bash
# Fetch pretrained backbone weights into checkpoints/pretrained/.
set -euo pipefail

DEST="checkpoints/pretrained"
mkdir -p "${DEST}"

echo "Put ViT-B/16 pretrained weights in ${DEST} (see docs/reproducibility.md)."
echo "Backbone is trained from scratch by default (model.pretrained=false)."
