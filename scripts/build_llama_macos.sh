#!/usr/bin/env bash
# Build llama.cpp from source for macOS (Apple Silicon, Metal) at the commit pinned in
# configs/engines.lock.json: llama-server, llama-cli and ggml-rpc-server with RPC enabled.
#
# Usage: scripts/build_llama_macos.sh [SRC_DIR] [--jobs N] [--check-only]
#   SRC_DIR        clone/checkout location (default: $HOME/.hypercompute/src/llama.cpp)
#   --jobs N       parallel build jobs (default: hw.logicalcpu)
#   --check-only   only run the prerequisite checks
# Env: HYPERCOMPUTE_CONFIG_DIR overrides the directory holding engines.lock.json.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
LOCK_FILE="${HYPERCOMPUTE_CONFIG_DIR:-$REPO_ROOT/configs}/engines.lock.json"
FALLBACK_COMMIT=39da130fcd1b6ca47d3c0522a40eccff41ea3dfb
FALLBACK_BUILD_NUMBER=11770
FALLBACK_REPO=https://github.com/ggml-org/llama.cpp
TARGETS="llama-server llama-cli ggml-rpc-server"
MIN_CMAKE=3.18

log() { printf '[hypercompute] %s\n' "$*" >&2; }
die() {
  printf '[hypercompute] ERROR: %s\n' "$*" >&2
  exit 1
}

SRC_DIR=''
JOBS=''
CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) [ $# -ge 2 ] || die "--jobs needs a value"; JOBS=$2; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h | --help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) [ -z "$SRC_DIR" ] || die "only one SRC_DIR may be given"; SRC_DIR=$1; shift ;;
  esac
done
[ "$(uname -s)" = Darwin ] || die "this script builds for macOS; use build_llama_linux.sh or build_llama_windows.ps1"
SRC_DIR=${SRC_DIR:-$HOME/.hypercompute/src/llama.cpp}
if [ -n "$JOBS" ]; then
  printf '%s' "$JOBS" | grep -Eq '^[1-9][0-9]*$' || die "--jobs must be a positive integer"
else
  JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
fi

# /usr/bin/python3 (and git, make, clang) are installer stubs until the Command Line Tools exist.
has_clt() { xcode-select -p >/dev/null 2>&1; }
python3_usable() {
  local py
  py=$(command -v python3 2>/dev/null) || return 1
  [ "$py" != /usr/bin/python3 ] || has_clt
}

# ------------------------------------------------------------------ pinned engine identity

# lock_get KEY: python3 reads llama_cpp.<KEY> (KEY dotted); without python, grep the pretty-printed lock.
lock_get() {
  local key=$1
  if python3_usable; then
    python3 -c 'import json, sys
node = json.load(open(sys.argv[1]))
for part in ["llama_cpp"] + sys.argv[2].split("."):
    node = node.get(part) if isinstance(node, dict) else None
print("" if node is None else node)' "$LOCK_FILE" "$key" 2>/dev/null || true
    return 0
  fi
  case $key in
    source_build.commit)
      # Join lines and cut everything up to the source_build object, so pretty and minified locks both work.
      { tr -d '\n\r' <"$LOCK_FILE" | sed 's/.*"source_build"[[:space:]]*:[[:space:]]*{//' |
        grep -Eo '"commit"[[:space:]]*:[[:space:]]*"[0-9a-f]{40}"' | grep -Eo '[0-9a-f]{40}' | head -n 1; } 2>/dev/null || true ;;
    build_number)
      { grep -Eo '"build_number"[[:space:]]*:[[:space:]]*[0-9]+' "$LOCK_FILE" | grep -Eo '[0-9]+$' | head -n 1; } 2>/dev/null || true ;;
    repo)
      { grep -Eo '"repo"[[:space:]]*:[[:space:]]*"https://[^"]+"' "$LOCK_FILE" | grep -Eo 'https://[^"]+' | head -n 1; } 2>/dev/null || true ;;
  esac
}

if [ -f "$LOCK_FILE" ]; then
  COMMIT=$(lock_get source_build.commit)
  BUILD_NUMBER=$(lock_get build_number)
  REPO_URL=$(lock_get repo)
  [ -n "$COMMIT" ] || die "$LOCK_FILE has no llama_cpp.source_build.commit"
  log "pinned commit $COMMIT from $LOCK_FILE"
else
  COMMIT=$FALLBACK_COMMIT
  BUILD_NUMBER=$FALLBACK_BUILD_NUMBER
  REPO_URL=$FALLBACK_REPO
  log "$LOCK_FILE not found; falling back to commit $COMMIT (b$BUILD_NUMBER)"
fi
REPO_URL=${REPO_URL:-$FALLBACK_REPO}
printf '%s' "$COMMIT" | grep -Eq '^[0-9a-f]{40}$' || die "pinned commit '$COMMIT' is not a 40-hex git sha"
printf '%s' "$REPO_URL" | grep -Eq '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$' ||
  die "repo '$REPO_URL' is not a github https URL"
if [ -n "$BUILD_NUMBER" ]; then
  printf '%s' "$BUILD_NUMBER" | grep -Eq '^[0-9]+$' || die "build_number '$BUILD_NUMBER' is not an integer"
fi

# ------------------------------------------------------------------ prerequisites

PROBLEMS=''
problem() { PROBLEMS="${PROBLEMS}  - $1"$'\n'; }

version_ge() {  # version_ge HAVE NEED (major.minor)
  local have_major=${1%%.*} have_rest=${1#*.} need_major=${2%%.*} need_minor=${2#*.}
  local have_minor=${have_rest%%.*}
  [ "$have_major" -gt "$need_major" ] || { [ "$have_major" -eq "$need_major" ] && [ "$have_minor" -ge "$need_minor" ]; }
}

if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
  problem "this shell runs under Rosetta (x86_64 emulation) and would build Intel binaries without Metal on Apple Silicon. Open a native terminal or run: arch -arm64 /bin/bash $0"
elif [ "$(uname -m)" != arm64 ]; then
  log "WARNING: $(uname -m) Mac. Hypercompute targets Apple Silicon; an Intel build is not validated."
fi
if has_clt; then
  git --version >/dev/null 2>&1 || problem "git does not run: reinstall the Command Line Tools (sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install)"
  command -v clang++ >/dev/null 2>&1 || problem "clang++ missing: xcode-select --install"
else
  problem "Xcode Command Line Tools missing (compiler, git, make): xcode-select --install"
fi
if command -v cmake >/dev/null 2>&1; then
  CMAKE_VERSION=$({ cmake --version | head -n 1 | grep -Eo '[0-9]+\.[0-9]+' | head -n 1; } || true)
  [ -n "$CMAKE_VERSION" ] && version_ge "$CMAKE_VERSION" "$MIN_CMAKE" ||
    problem "cmake ${CMAKE_VERSION:-?} is older than $MIN_CMAKE: brew upgrade cmake"
elif command -v brew >/dev/null 2>&1; then
  problem "cmake missing: brew install cmake"
else
  problem "cmake missing: install Homebrew (https://brew.sh) then run brew install cmake, or install CMake.app from https://cmake.org/download/ and run: sudo \"/Applications/CMake.app/Contents/bin/cmake-gui\" --install"
fi

if [ -n "$PROBLEMS" ]; then
  printf '[hypercompute] cannot build yet. Fix these first:\n%s' "$PROBLEMS" >&2
  exit 2
fi
log "prerequisites ok (accelerator: metal, jobs: $JOBS)"
[ "$CHECK_ONLY" -eq 0 ] || exit 0

# ------------------------------------------------------------------ checkout

if [ -e "$SRC_DIR" ] && [ ! -d "$SRC_DIR/.git" ] && [ -n "$(ls -A "$SRC_DIR" 2>/dev/null)" ]; then
  die "$SRC_DIR exists and is not a git checkout; pass another SRC_DIR"
fi
mkdir -p "$(dirname "$SRC_DIR")"
SRC_DIR="$(cd "$(dirname "$SRC_DIR")" && pwd)/$(basename "$SRC_DIR")"
if [ ! -d "$SRC_DIR/.git" ]; then
  log "cloning $REPO_URL into $SRC_DIR (blobless clone, full commit history)"
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$SRC_DIR"
else
  if [ -n "$(git -C "$SRC_DIR" status --porcelain --untracked-files=no)" ]; then
    die "$SRC_DIR has local modifications to tracked files; commit or stash them first"
  fi
fi
if ! git -C "$SRC_DIR" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  log "fetching $COMMIT"
  git -C "$SRC_DIR" fetch origin "$COMMIT"
fi
git -C "$SRC_DIR" checkout --detach "$COMMIT"
[ "$(git -C "$SRC_DIR" rev-parse HEAD)" = "$COMMIT" ] || die "checkout did not land on $COMMIT"
# git checkout exits 0 even when it could not write some files; never build a partial tree.
DIRTY=$(git -C "$SRC_DIR" status --porcelain --untracked-files=no)
if [ -n "$DIRTY" ]; then
  die "git checkout of $COMMIT left tracked files missing or modified ($(printf '%s\n' "$DIRTY" | wc -l | tr -d ' '), first: '$(printf '%s\n' "$DIRTY" | head -n 1)'). Fix the cause git printed above (disk space, permissions, a file name clash on a case-insensitive volume, which APFS is by default), then delete $SRC_DIR and rerun."
fi

# ------------------------------------------------------------------ configure + build

BUILD_DIR="$SRC_DIR/build-hypercompute"
CMAKE_ARGS=(-S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=OFF)
# Same version banner as the official release ("build 11770, commit 39da130fc"), so engine identity
# checks match prebuilt peers.
if [ -n "$BUILD_NUMBER" ]; then
  CMAKE_ARGS+=("-DLLAMA_BUILD_NUMBER=$BUILD_NUMBER" "-DLLAMA_BUILD_COMMIT=${COMMIT:0:9}")
fi

log "cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}" || die "cmake configure failed (see above). Make sure the Command Line Tools are current: softwareupdate --list"
# shellcheck disable=SC2086 # TARGETS is a fixed space-separated list
cmake --build "$BUILD_DIR" --config Release -j "$JOBS" --target $TARGETS ||
  die "build failed (see the compiler output above)"

# ------------------------------------------------------------------ verify

BIN_DIR="$BUILD_DIR/bin"
for target in $TARGETS; do
  [ -x "$BIN_DIR/$target" ] || die "expected $BIN_DIR/$target after the build"
done
log "llama-server --version:"
VERSION_OUT=$("$BIN_DIR/llama-server" --version 2>&1) || die "llama-server --version failed: $VERSION_OUT"
printf '%s\n' "$VERSION_OUT"
if [ -n "$BUILD_NUMBER" ] && ! printf '%s' "$VERSION_OUT" | grep -q "build $BUILD_NUMBER, commit ${COMMIT:0:9}"; then
  log "WARNING: version banner does not say 'build $BUILD_NUMBER, commit ${COMMIT:0:9}'"
fi
log "llama-server --list-devices (expect an MTL0 line on Apple Silicon):"
"$BIN_DIR/llama-server" --list-devices || log "WARNING: --list-devices failed"

printf '\nBuilt %s (metal) at commit %s.\nBinaries: %s\nUse them with:\n  export HYPERCOMPUTE_LLAMA_BIN_DIR=%q\n' \
  "$TARGETS" "$COMMIT" "$BIN_DIR" "$BIN_DIR"
