#!/bin/bash
# Run on a Mac over SSH as the hypercompute user. Installs or updates the code from the bundle the G14 copied to
# ~/hypercompute.bundle, then the pinned llama.cpp engine. Safe to run again.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Use the standalone command line tools when present: full Xcode refuses git until its license is accepted (sudo).
if [ -d /Library/Developer/CommandLineTools ]; then export DEVELOPER_DIR=/Library/Developer/CommandLineTools; fi
BUNDLE="$HOME/hypercompute.bundle"
REPO="${HC_REPO:-$HOME/hc-peer}"
BRANCH="codex/iroh-transport"

command -v git >/dev/null || { echo "git missing: run xcode-select --install from an admin account"; exit 2; }
command -v uv >/dev/null || { echo "uv missing: run the uv installer as this user"; exit 2; }
[ -f "$BUNDLE" ] || { echo "missing $BUNDLE"; exit 2; }

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch "$BUNDLE" "$BRANCH:refs/remotes/bundle/$BRANCH"
  git -C "$REPO" checkout -B "$BRANCH" "refs/remotes/bundle/$BRANCH"
else
  git clone --branch "$BRANCH" "$BUNDLE" "$REPO"
fi
cd "$REPO"
git log --oneline -1
uv sync
uv run hypercompute engine install --yes
uv run hypercompute doctor || true
df -h "$HOME" | tail -1
sysctl -n hw.memsize | awk '{printf "ram_gib %.1f\n", $1/1073741824}'
