#!/usr/bin/env bash
# Log in to GitHub with a device code (no browser on the server, no token to paste).
#
# Runs `gh auth login --web` inside tmux so it survives the ssh session, answers the
# "Authenticate Git with your GitHub credentials?" prompt, then prints the one-time code.
# You finish the login in a browser: https://github.com/login/device
#
# Usage:
#   bash tools/gh-login-device.sh          # start (or restart) the login
#   bash tools/gh-login-device.sh --check  # show the current pane / auth status
set -euo pipefail

GH_ROOT="/root/autodl-tmp/lab/tools/gh"
GH_VERSION="2.102.0"
GH_BIN="${GH_ROOT}/gh_${GH_VERSION}_linux_amd64/bin/gh"
export GH_CONFIG_DIR="${GH_ROOT}/config"        # keeps the token off the system disk
SESSION="${GH_SESSION:-ghlogin}"

if [ "${1:-}" = "--check" ]; then
  tmux capture-pane -pt "${SESSION}" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -8 || true
  echo "--- auth status ---"
  "${GH_BIN}" auth status 2>&1 | head -8
  exit 0
fi

[ -x "${GH_BIN}" ] || { echo "gh not found at ${GH_BIN} — run tools/install-gh.sh first" >&2; exit 1; }
mkdir -p "${GH_CONFIG_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -x 200 -y 50
tmux send-keys -t "${SESSION}" \
  "export GH_CONFIG_DIR='${GH_CONFIG_DIR}'; '${GH_BIN}' auth login --hostname github.com --git-protocol https --web" Enter

sleep 2                                              # answer the git-credential question
tmux send-keys -t "${SESSION}" "Y" Enter
sleep 3

echo "=== device code ==="
tmux capture-pane -pt "${SESSION}" | grep -v '^[[:space:]]*$' | tail -8
echo
echo "Now open https://github.com/login/device and enter the code above."
echo "Then run: bash tools/gh-login-device.sh --check"
