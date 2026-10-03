#!/usr/bin/env bash
# Install GitHub CLI into the persistent data disk and wire it up for this repo.
#
# Why a script: everything under /root (system disk) is wiped when the AutoDL
# container is reset, so gh, its auth token and the git credential helper are all
# kept on /root/autodl-tmp and restored with this single command.
#
# Usage:  bash tools/install-gh.sh
set -euo pipefail

GH_ROOT="/root/autodl-tmp/lab/tools/gh"
GH_VERSION="2.102.0"
GH_BIN="${GH_ROOT}/gh_${GH_VERSION}_linux_amd64/bin/gh"
GH_CONFIG_DIR="${GH_ROOT}/config"          # keeps hosts.yml off the system disk
LINK_DIR="/root/autodl-tmp/lab/tools/bin"

echo "[1/4] gh binary"
if [ -x "${GH_BIN}" ]; then
  echo "      already installed: $(basename "$(dirname "${GH_BIN}")")"
else
  mkdir -p "${GH_ROOT}"
  curl -sL --retry 2 --max-time 240 -o "${GH_ROOT}/gh.tar.gz" \
    "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz"
  tar -xzf "${GH_ROOT}/gh.tar.gz" -C "${GH_ROOT}"
  rm -f "${GH_ROOT}/gh.tar.gz"
fi
"${GH_BIN}" --version

echo "[2/4] expose on PATH -> ${LINK_DIR}/gh"
mkdir -p "${LINK_DIR}"
ln -sf "${GH_BIN}" "${LINK_DIR}/gh"

echo "[3/4] persistent config dir -> ${GH_CONFIG_DIR}"
mkdir -p "${GH_CONFIG_DIR}"

echo "[4/5] shell wiring (GH_CONFIG_DIR + PATH)"
if ! grep -q "GH_CONFIG_DIR.*lab/tools/gh" /root/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "# BEGIN GitHub CLI (added by tools/install-gh.sh)"
    echo "export GH_CONFIG_DIR=\"${GH_CONFIG_DIR}\""
    echo "export PATH=\"${LINK_DIR}:\$PATH\""
    echo "# END GitHub CLI"
  } >> /root/.bashrc
  echo "      appended block to /root/.bashrc"
else
  echo "      already wired in /root/.bashrc"
fi

echo "[5/5] git transport"
# GitHub over the direct route fails intermittently ("Error in the HTTP2 framing
# layer"), so route git through the local proxy and speak HTTP/1.1.
git config --global http.version HTTP/1.1
git config --global http.proxy http://127.0.0.1:6666
echo "      http.version=$(git config --global --get http.version) " \
     "http.proxy=$(git config --global --get http.proxy)"

cat <<EOF

Installed.
  binary : ${GH_BIN}
  config : ${GH_CONFIG_DIR}   (auth token lives here, survives container reset)
  PATH   : ${LINK_DIR}/gh

Next:
  export GH_CONFIG_DIR="${GH_CONFIG_DIR}"
  export PATH="${LINK_DIR}:\$PATH"
  gh auth login --hostname github.com --git-protocol https --web
  gh auth setup-git          # makes plain 'git push' use the gh credential helper
EOF
