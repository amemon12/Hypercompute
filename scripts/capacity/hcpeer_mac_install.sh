#!/usr/bin/env bash
# Make an hcpeer engine on a macOS peer, as the normal user, without touching the stock engine.
#
# Clones the installed llama.cpp bin dir (the one holding ggml-rpc-server) to a sibling "<dir>-hcpeer", puts
# native/hcpeer/build/libggml-rpc.dylib in the clone once its sha256 matches libggml-rpc.dylib.sha256, re-signs it
# ad hoc only if its signature does not verify, clears quarantine attributes, then starts the clone's ggml-rpc-server
# once with HC_PEER_MMAP=1 and -c and requires the hcpeer v4 banner before stopping it. Prints the clone path on
# stdout (everything else goes to stderr) and exits non-zero on any failure. Running it again re-checks an up-to-date
# clone and rebuilds a stale one. Details: docs/research/hcpeer-macos.md, "Running it".
#
#   bash scripts/capacity/hcpeer_mac_install.sh [--engine-dir <stock bin dir>] [--dylib <libggml-rpc.dylib>]
#
# Then join with HYPERCOMPUTE_LLAMA_BIN_DIR=<printed path> and HC_PEER_MMAP=1.
# Env: HYPERCOMPUTE_HOME (default ~/.hypercompute), HYPERCOMPUTE_CONFIG_DIR (holds engines.lock.json).
set -euo pipefail

PROG=hcpeer_mac_install
SERVER=ggml-rpc-server
RPC_LINK=libggml-rpc.dylib     # the library name this script replaces
RPC_LOADED=libggml-rpc.0.dylib # the @rpath name libggml loads; must be the same file
RECORD=.hcpeer-clone           # key=value lines saying what the clone was made from
ENGINE_MARKER=.hypercompute-engine.json
MIN_VERSION=4
SMOKE_TIMEOUT_S=30
SMOKE_PORTS="47152 47153 47154 47155 47156"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENGINE_DIR=''
DYLIB="$REPO/native/hcpeer/build/libggml-rpc.dylib"
CLONE=''
STAGING=''
SMOKE_PID=''
SMOKE_TMP=''

say() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() {
  printf '%s: error: %s\n' "$PROG" "$*" >&2
  exit 1
}

stop_smoke_server() {
  if [ -n "$SMOKE_PID" ]; then
    kill "$SMOKE_PID" 2>/dev/null || true
    wait "$SMOKE_PID" 2>/dev/null || true
    SMOKE_PID=''
  fi
}

cleanup() {
  stop_smoke_server
  if [ -n "$SMOKE_TMP" ]; then rm -rf "$SMOKE_TMP"; fi
  if [ -n "$STAGING" ]; then rm -rf "$STAGING"; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' >&2; }

parse_args() {
  while [ $# -gt 0 ]; do
    case $1 in
      --engine-dir)
        [ $# -ge 2 ] || die "--engine-dir needs the stock bin dir"
        ENGINE_DIR=$2
        shift 2
        ;;
      --dylib)
        [ $# -ge 2 ] || die "--dylib needs a file"
        DYLIB=$2
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *) die "unknown argument '$1' (see --help)" ;;
    esac
  done
}

check_host() {
  local tool
  [ "$(uname -s)" = Darwin ] || die "this runs on the Mac peer, not on $(uname -s)"
  [ "$(uname -m)" = arm64 ] || die "the hcpeer library is arm64 only, this Mac is $(uname -m)"
  for tool in codesign xattr cp mv cmp find readlink mktemp; do
    command -v "$tool" >/dev/null 2>&1 || die "missing the macOS tool '$tool'"
  done
}

sha256_of() {
  local out
  if command -v shasum >/dev/null 2>&1; then out=$(shasum -a 256 "$1"); else out=$(sha256sum "$1"); fi
  printf '%s\n' "${out%% *}"
}

# The bin dir `hypercompute engine install` made (build_hcpeer_mac.py find_engine_dir reads it the same way).
default_engine_dir() {
  local home=${HYPERCOMPUTE_HOME:-$HOME/.hypercompute} release root sub
  local lock="${HYPERCOMPUTE_CONFIG_DIR:-$REPO/configs}/engines.lock.json"
  case $home in "~"/*) home="$HOME/${home#\~/}" ;; esac
  [ -f "$lock" ] || die "missing $lock: run this from the repo checkout"
  release=$(sed -n 's/.*"release": *"\(b[0-9][0-9]*\)".*/\1/p' "$lock" | head -n 1)
  [ -n "$release" ] || die "cannot read the llama.cpp release from $lock"
  root="$home/engines/llama-$release-macos-arm64"
  [ -f "$root/$ENGINE_MARKER" ] ||
    die "no installed engine at $root: run 'uv run hypercompute engine install' or pass --engine-dir"
  sub=$(sed -n 's/.*"bin_subdir": *"\([^"]*\)".*/\1/p' "$root/$ENGINE_MARKER" | head -n 1)
  case $sub in /* | *..*) die "unexpected bin_subdir '$sub' in $root/$ENGINE_MARKER" ;; esac
  if [ -n "$sub" ]; then printf '%s/%s\n' "$root" "$sub"; else printf '%s\n' "$root"; fi
}

# The real file NAME resolves to inside DIR, following only same-dir relative symlinks, so it never leaves DIR.
real_name() {
  local dir=$1 name=$2 hops=0 target
  while [ -L "$dir/$name" ]; do
    target=$(readlink "$dir/$name") || return 1
    case $target in '' | */*) return 1 ;; esac
    name=$target
    hops=$((hops + 1))
    [ "$hops" -le 8 ] || return 1
  done
  [ -f "$dir/$name" ] || return 1
  printf '%s\n' "$name"
}

# The real libggml-rpc file in DIR; libggml-rpc.0.dylib, when present, must be that same file.
rpc_real_name() {
  local dir=$1 real loaded
  real=$(real_name "$dir" "$RPC_LINK") || die "$dir/$RPC_LINK is missing or links outside its directory"
  if [ -e "$dir/$RPC_LOADED" ] || [ -L "$dir/$RPC_LOADED" ]; then
    loaded=$(real_name "$dir" "$RPC_LOADED") || die "$dir/$RPC_LOADED is broken or links outside its directory"
    [ "$loaded" = "$real" ] || die "$dir/$RPC_LOADED ($loaded) is not $RPC_LINK ($real): unexpected engine layout"
  fi
  printf '%s\n' "$real"
}

check_inputs() {
  local build_json first
  [ -f "$DYLIB" ] || die "missing $DYLIB (it comes with the repo checkout; see docs/research/hcpeer-macos.md)"
  [ -f "$DYLIB.sha256" ] || die "missing $DYLIB.sha256"
  first=$(head -n 1 "$DYLIB.sha256" | tr -d '\r')
  EXPECTED=${first%% *}
  case $EXPECTED in '' | *[!0-9a-f]*) die "$DYLIB.sha256 does not start with a sha256" ;; esac
  [ "${#EXPECTED}" -eq 64 ] || die "$DYLIB.sha256 does not start with a sha256"
  [ "$(sha256_of "$DYLIB")" = "$EXPECTED" ] || die "$DYLIB does not match $DYLIB.sha256: pull again or rebuild it"
  build_json="$(dirname "$DYLIB")/build.json"
  [ -f "$build_json" ] || die "missing $build_json (it names the release library hcpeer was built against)"
  ORIGINAL=$(sed -n 's/.*"release_original_sha256": *"\([0-9a-f]\{64\}\)".*/\1/p' "$build_json" | head -n 1)
  [ -n "$ORIGINAL" ] || die "no release_original_sha256 in $build_json"
}

check_stock() {
  if [ -z "$ENGINE_DIR" ]; then ENGINE_DIR=$(default_engine_dir) || exit 1; fi
  [ -d "$ENGINE_DIR" ] || die "no engine dir at $ENGINE_DIR"
  ENGINE_DIR=$(cd "$ENGINE_DIR" && pwd -P)
  [ -x "$ENGINE_DIR/$SERVER" ] || die "$ENGINE_DIR has no $SERVER: pass the engine bin dir with --engine-dir"
  [ ! -e "$ENGINE_DIR/$RECORD" ] || die "$ENGINE_DIR is an hcpeer clone; pass the stock engine dir"
  STOCK_RPC=$(rpc_real_name "$ENGINE_DIR") || exit 1
  STOCK_RPC_SHA=$(sha256_of "$ENGINE_DIR/$STOCK_RPC")
  if [ "$STOCK_RPC_SHA" = "$EXPECTED" ]; then
    die "$ENGINE_DIR already has hcpeer in place (build_hcpeer_mac.py install); restore the stock library" \
      "first: uv run --no-sync python scripts/build_hcpeer_mac.py rollback"
  fi
  [ "$STOCK_RPC_SHA" = "$ORIGINAL" ] ||
    die "$ENGINE_DIR/$STOCK_RPC is not the release library hcpeer was built against" \
      "(sha256 $STOCK_RPC_SHA, expected $ORIGINAL)"
  STOCK_SERVER_SHA=$(sha256_of "$ENGINE_DIR/$SERVER")
  CLONE="$ENGINE_DIR-hcpeer"
}

record_value() { sed -n "s/^$2=//p" "$1/$RECORD" | head -n 1; }

clone_is_current() {
  local real
  [ "$(record_value "$CLONE" source)" = "$ENGINE_DIR" ] || return 1
  [ "$(record_value "$CLONE" source_rpc_sha256)" = "$STOCK_RPC_SHA" ] || return 1
  [ "$(record_value "$CLONE" source_server_sha256)" = "$STOCK_SERVER_SHA" ] || return 1
  [ "$(record_value "$CLONE" hcpeer_sha256)" = "$EXPECTED" ] || return 1
  real=$(rpc_real_name "$CLONE" 2>/dev/null) || return 1
  [ "$real" = "$(record_value "$CLONE" installed_file)" ] || return 1
  [ "$(sha256_of "$CLONE/$real")" = "$(record_value "$CLONE" installed_sha256)" ] || return 1
  cmp -s "$ENGINE_DIR/$SERVER" "$CLONE/$SERVER"
}

ensure_signed() {
  if codesign --verify "$1" 2>/dev/null; then
    return 0
  fi
  say "re-signing $(basename "$1") ad hoc: its signature does not verify"
  codesign -s - -f "$1" >&2 || die "codesign could not sign $1"
  codesign --verify "$1" >&2 || die "$1 still fails codesign --verify after signing"
}

clear_quarantine() {
  local f
  find "$1" \( -type f -o -type d \) -print | while IFS= read -r f; do
    if xattr -p com.apple.quarantine "$f" >/dev/null 2>&1; then
      xattr -d com.apple.quarantine "$f" || die "cannot clear the quarantine attribute of $f"
    fi
  done
}

# Copy the stock dir to STAGING (an APFS clone when cp -c works) and put the hcpeer library in it.
build_staging() {
  local real sha
  STAGING="$CLONE.staging.$$"
  rm -rf "$STAGING"
  say "cloning $ENGINE_DIR"
  if ! cp -cR "$ENGINE_DIR" "$STAGING" 2>/dev/null; then
    rm -rf "$STAGING"
    cp -R "$ENGINE_DIR" "$STAGING" || die "cannot copy $ENGINE_DIR to $STAGING"
  fi
  rm -f "$STAGING/$ENGINE_MARKER" # a clone must never be picked up as the installed engine
  real=$(rpc_real_name "$STAGING") || exit 1
  cp "$DYLIB" "$STAGING/$real.hcpeer-tmp" || die "cannot copy $DYLIB into the clone"
  sha=$(sha256_of "$STAGING/$real.hcpeer-tmp")
  [ "$sha" = "$EXPECTED" ] || die "the copy of $DYLIB in the clone has sha256 $sha, not $EXPECTED"
  # a rename gives the library a new inode: macOS kills a process that maps a signed file rewritten in place
  mv -f "$STAGING/$real.hcpeer-tmp" "$STAGING/$real" || die "cannot replace $real in the clone"
  ensure_signed "$STAGING/$real"
  clear_quarantine "$STAGING"
  INSTALLED_FILE=$real
}

write_record() {
  {
    printf 'source=%s\n' "$ENGINE_DIR"
    printf 'source_rpc_sha256=%s\n' "$STOCK_RPC_SHA"
    printf 'source_server_sha256=%s\n' "$STOCK_SERVER_SHA"
    printf 'hcpeer_sha256=%s\n' "$EXPECTED"
    printf 'installed_file=%s\n' "$INSTALLED_FILE"
    printf 'installed_sha256=%s\n' "$(sha256_of "$STAGING/$INSTALLED_FILE")"
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$STAGING/$RECORD" || die "cannot write $STAGING/$RECORD"
}

swap_in() {
  local old=''
  if [ -e "$CLONE" ]; then
    old="$CLONE.old.$$"
    mv "$CLONE" "$old" || die "cannot move the old clone aside"
  fi
  if ! mv "$STAGING" "$CLONE"; then
    if [ -n "$old" ]; then mv "$old" "$CLONE" || true; fi
    die "cannot move the new clone to $CLONE"
  fi
  STAGING=''
  if [ -n "$old" ]; then rm -rf "$old"; fi
}

# 0 once the server printed its transport line, bound its port and still runs.
wait_smoke_server() {
  local log=$1 i=0
  while [ "$i" -lt $((SMOKE_TIMEOUT_S * 5)) ]; do
    if grep -q '^ *transport *:' "$log"; then
      sleep 1
      if grep -q 'Failed to create server socket' "$log"; then return 1; fi
      kill -0 "$SMOKE_PID" 2>/dev/null
      return
    fi
    kill -0 "$SMOKE_PID" 2>/dev/null || return 1
    sleep 0.2
    i=$((i + 1))
  done
  return 1
}

check_banner() {
  local log=$1 line version
  line=$(grep -m 1 'hcpeer mmap' "$log" || true)
  [ -n "$line" ] ||
    die "the clone's $SERVER printed no hcpeer banner, so it did not load the hcpeer library:" \
      "$(tail -n 15 "$log")"
  case $line in *': on (v'*) ;; *) die "hcpeer is not on in the clone's $SERVER: $line" ;; esac
  version=${line#*: on (v}
  version=${version%%,*}
  case $version in '' | *[!0-9]*) die "cannot read the hcpeer version from: $line" ;; esac
  [ "$version" -ge "$MIN_VERSION" ] ||
    die "the clone runs hcpeer v$version; v$MIN_VERSION or later is needed (pull the current native/hcpeer/build)"
  say "smoke passed: hcpeer v$version on"
}

# Start DIR's ggml-rpc-server on 127.0.0.1 with -d CPU -c and HC_PEER_MMAP=1 in a scratch cache, check the banner,
# stop it. DYLD_* are unset so the clone loads only its own libraries.
smoke() {
  local dir=$1 log port started=''
  SMOKE_TMP=$(mktemp -d "${TMPDIR:-/tmp}/hcpeer-smoke.XXXXXX") || die "mktemp failed"
  log="$SMOKE_TMP/server.log"
  for port in $SMOKE_PORTS; do
    env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES -u HC_PEER_CACHE_MAX_GIB \
      HC_PEER_MMAP=1 HC_PEER_CACHE_MIN_FREE_GIB=0 LLAMA_CACHE="$SMOKE_TMP/cache" \
      "$dir/$SERVER" -H 127.0.0.1 -p "$port" -d CPU -c >"$log" 2>&1 </dev/null &
    SMOKE_PID=$!
    if wait_smoke_server "$log"; then
      started=$port
      break
    fi
    stop_smoke_server
    grep -q 'Failed to create server socket' "$log" ||
      die "the clone's $SERVER did not start: $(tail -n 15 "$log")"
  done
  [ -n "$started" ] || die "the clone's $SERVER found no free port among $SMOKE_PORTS"
  stop_smoke_server
  check_banner "$log"
  rm -rf "$SMOKE_TMP"
  SMOKE_TMP=''
}

main() {
  parse_args "$@"
  check_host
  check_inputs
  check_stock
  if [ -e "$CLONE" ] && [ ! -f "$CLONE/$RECORD" ]; then
    die "$CLONE exists and was not made by this script: move it away first"
  fi
  if [ -e "$CLONE" ] && clone_is_current; then
    say "$CLONE is up to date, checking it again"
    smoke "$CLONE"
  else
    build_staging
    smoke "$STAGING"
    write_record
    swap_in
    say "made $CLONE; the stock engine at $ENGINE_DIR is unchanged"
  fi
  printf '%s\n' "$CLONE"
}

main "$@"
