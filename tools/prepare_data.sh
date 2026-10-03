#!/usr/bin/env bash
# Build dataset manifests and verify the data mount.
set -euo pipefail

python - <<'PY'
from pathlib import Path

root = Path("data")
for name in ("LasHeR", "RGBT234"):
    p = root / name
    print(f"{name:9s} present={p.exists()}  -> {p.resolve() if p.exists() else 'MISSING'}")
PY
