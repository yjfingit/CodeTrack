#!/usr/bin/env bash
# Install the repository git hooks (currently: post-commit auto-push).
# Idempotent — safe to re-run after a fresh clone or a container reset.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="${REPO_ROOT}/tools/git-hooks"
HOOK_DST="$(git rev-parse --git-dir)/hooks"

if [ ! -d "${HOOK_SRC}" ]; then
  echo "[hooks] ${HOOK_SRC} not found" >&2
  exit 1
fi

mkdir -p "${HOOK_DST}"

for hook in "${HOOK_SRC}"/*; do
  name="$(basename "${hook}")"
  # skip the hook for our own hook files? no — post-commit is the only one
  cp -f "${hook}" "${HOOK_DST}/${name}"
  chmod +x "${HOOK_DST}/${name}"
  echo "[hooks] installed ${name} -> ${HOOK_DST}/${name}"
done

echo "[hooks] done. Auto-push log: $(git rev-parse --git-dir)/push.log"
