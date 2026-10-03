#!/usr/bin/env bash
# Manual fallback for GitHub sync — stage everything, commit if needed, push.
#
# Usage:
#   bash tools/sync-to-github.sh                  # message: "chore: sync"
#   bash tools/sync-to-github.sh "feat: add codec"
set -euo pipefail

MSG="${1:-chore: sync}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[sync] not a git repository" >&2
  exit 1
fi

echo "[sync] remote: $(git remote get-url origin 2>/dev/null || echo '(none)')"

git add -A

if git diff --cached --quiet; then
  echo "[sync] nothing to commit"
else
  git commit -m "${MSG}"
fi

GIT_TERMINAL_PROMPT=0 git push origin HEAD
echo "[sync] pushed $(git rev-parse --short HEAD) to origin/$(git branch --show-current)"
