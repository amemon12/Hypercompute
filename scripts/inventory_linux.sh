#!/usr/bin/env bash
# Hypercompute raw hardware inventory for Linux (x86_64 or arm64, including GB10 / DGX Spark class boxes).
#
# Read-only. Prints exactly one JSON document on stdout:
#   {"schema": 1, "platform": "linux", "script": "inventory_linux.sh", "collected_at": "<UTC ISO-8601>",
#    "probes": {"<name>": {"cmd": [...], "exit_code": <int|null>, "stdout": "...", "stderr": "...",
#                          "error": "..." (only when the tool is missing or timed out)}}}
# A missing tool never aborts the document: its probe gets exit_code null and an error string.
# Values whose key mentions serial/uuid/udid and anything shaped like a MAC address are redacted.
# NUL bytes and invalid UTF-8 in probe output are dropped, identically for both JSON writers.
#
# Environment knobs:
#   HC_INVENTORY_JSON=auto|python|bash  JSON writer (auto = python3 when usable, else the bash escaper)
#   HC_PYTHON=<path>                     python interpreter for the python writer
#   HC_PROBE_TIMEOUT_S=<seconds>         per-probe timeout (default 45)
set -euo pipefail
export LC_ALL=C

PLATFORM=linux
SCRIPT_NAME=inventory_linux.sh
TIMEOUT_S=${HC_PROBE_TIMEOUT_S:-45}
case $TIMEOUT_S in '' | *[!0-9]*) echo "HC_PROBE_TIMEOUT_S must be a whole number of seconds" >&2; exit 2 ;; esac

TMP_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t hcinv)
trap 'rm -rf "$TMP_DIR"' EXIT
PROBE_COUNT=0

# ------------------------------------------------------------------ probe runner (shared with macOS)

# Polls instead of using a sleeping watchdog process: no stray child keeps stdout open, and it needs
# neither GNU timeout (absent on macOS) nor signal delivery to a waiting subshell.
run_with_timeout() {
  local base=$1
  shift
  local rc=0 ticks=0 limit=$((TIMEOUT_S * 10))
  "$@" </dev/null >"$base.out" 2>"$base.err" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$ticks" -eq "$limit" ]; then
      printf '%s' "timed out after ${TIMEOUT_S}s" >"$base.error"
      kill -TERM "$pid" 2>/dev/null || true
    elif [ "$ticks" -ge $((limit + 20)) ]; then
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 0.1
    ticks=$((ticks + 1))
  done
  wait "$pid" || rc=$?
  printf '%s' "$rc" >"$base.rc"
}

# probe NAME CMD [ARG...]  -- runs CMD read-only and records cmd/exit_code/stdout/stderr.
probe() {
  local name=$1
  shift
  local base="$TMP_DIR/$PROBE_COUNT"
  PROBE_COUNT=$((PROBE_COUNT + 1))
  printf '%s' "$name" >"$base.name"
  printf '%s\n' "$@" >"$base.cmd"
  : >"$base.out"
  : >"$base.err"
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'null' >"$base.rc"
    printf '%s' "not found: $1" >"$base.error"
    return 0
  fi
  run_with_timeout "$base" "$@"
}

# clean_file SRC DST [redact] -- DST = SRC as valid UTF-8 without NUL bytes, so both JSON writers see the
# same text: NUL (bash command substitution drops it with a warning) and invalid UTF-8 are dropped.
# iconv -c alone is not enough: glibc's keeps 4-6 byte sequences above U+10FFFF, which UTF8_SED removes.
# Without iconv only ASCII is kept. With "redact", identifiers are removed before anything leaves this
# machine: JSON values whose key mentions serial/uuid/udid (system_profiler -json) and MAC addresses
# (ifconfig, networksetup, ip link). The MAC rule runs twice: each match consumes its trailing
# separator, so in "M1 M2 M3" one pass skips M2.
HAVE_ICONV=0
if command -v iconv >/dev/null 2>&1; then HAVE_ICONV=1; fi
UTF8_SED="$TMP_DIR/utf8.sed"
write_utf8_sed() {
  # F4 90..F4 BF and F5..FF lead bytes, each with the continuation bytes that follow it.
  printf 's/%b[%b-%b][%b-%b]*//g\n' '\0364' '\0220' '\0277' '\0200' '\0277' >"$UTF8_SED"
  printf 's/[%b-%b][%b-%b]*//g\n' '\0365' '\0377' '\0200' '\0277' >>"$UTF8_SED"
}

clean_file() {
  local src=$1 dst=$2 redact=${3:-}
  if [ ! -s "$src" ]; then
    : >"$dst"
    return 0
  fi
  if [ "$HAVE_ICONV" = 1 ]; then
    tr -d '\000' <"$src" >"$dst.nonul"
    # -c drops invalid input; GNU libiconv then exits 1 (glibc exits 0). Any other status is a failure.
    iconv -f UTF-8 -t UTF-8 -c <"$dst.nonul" >"$dst.utf8" 2>/dev/null || [ "$?" -eq 1 ]
  else
    tr -d '\000\200-\377' <"$src" >"$dst.utf8"
  fi
  if [ -z "$redact" ]; then
    sed -E -f "$UTF8_SED" "$dst.utf8" >"$dst"
    return 0
  fi
  sed -E -f "$UTF8_SED" \
    -e 's/("[^"]*([Ss][Ee][Rr][Ii][Aa][Ll]|[Uu][Uu][Ii][Dd]|[Uu][Dd][Ii][Dd])[^"]*"[[:space:]]*:[[:space:]]*)("[^"]*"|[0-9A-Fa-fx]+)/\1"[redacted]"/g' \
    -e 's/(^|[^0-9A-Fa-f:])([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}([^0-9A-Fa-f:]|$)/\1xx:xx:xx:xx:xx:xx\3/g' \
    -e 's/(^|[^0-9A-Fa-f:])([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}([^0-9A-Fa-f:]|$)/\1xx:xx:xx:xx:xx:xx\3/g' \
    "$dst.utf8" >"$dst"
}

# ------------------------------------------------------------------ JSON writers

PY_WRITER='
import json, os, sys
tmp, count, platform, script, collected_at = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]

def read(path):
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return None

probes = {}
for i in range(count):
    base = os.path.join(tmp, str(i))
    rc = read(base + ".rc")
    entry = {
        # one argument per line, split on "\n" only, exactly as the bash writer reads them
        "cmd": (read(base + ".cmd.clean") or "").split("\n")[:-1],
        "exit_code": None if rc in (None, "null") else int(rc),
        "stdout": read(base + ".out.clean") or "",
        "stderr": read(base + ".err.clean") or "",
    }
    error = read(base + ".error")
    if error is not None:
        entry["error"] = error
    probes[read(base + ".name")] = entry

doc = {"schema": 1, "platform": platform, "script": script, "collected_at": collected_at, "probes": probes}
json.dump(doc, sys.stdout, indent=1, ensure_ascii=True)
sys.stdout.write("\n")
'

# Pure-bash fallback writer. Escaping is done by sed (bytewise under LC_ALL=C, so UTF-8 passes through)
# rather than by bash pattern substitution, which mishandles \001 in some bash versions and is slow on
# large strings. Newlines are the only escape left to bash. One sed run escapes every field file and
# builtin reads load them: process creation, not escaping, dominates the run time on Git Bash/Cygwin.
ESCAPE_SED="$TMP_DIR/escape.sed"
write_escape_sed() {
  local pair
  {
    printf '%s\n' 's/\\/\\\\/g' 's/"/\\"/g'
    printf 's/%b/\\\\%s/g\n' '\t' t '\r' r '\b' b '\f' f
    # octal:hex for the remaining C0 controls and DEL (%b takes \0nnn octal in every bash version)
    for pair in 001:01 002:02 003:03 004:04 005:05 006:06 007:07 013:0b 016:0e 017:0f 020:10 021:11 \
      022:12 023:13 024:14 025:15 026:16 027:17 030:18 031:19 032:1a 033:1b 034:1c 035:1d 036:1e 037:1f \
      177:7f; do
      printf 's/%b/\\\\u00%s/g\n' "\\0${pair%:*}" "${pair#*:}"
    done
  } >"$ESCAPE_SED"
}

# json_str PATH -> REPLY = an escaped field file (see write_json_bash) as a JSON string literal.
json_str() {
  local content=''
  if [ -s "$1" ]; then
    IFS= read -r -d '' content <"$1" || true  # no NUL is left, so this reads to EOF
    content=${content%$'\n'}
    content=${content//$'\n'/\\n}
  fi
  REPLY="\"$content\""
}

write_json_bash() {
  local collected_at=$1 i base f sep arg_sep line rc=''
  local files=()
  write_escape_sed
  # Escape all field files in place with one sed run. A newline goes onto each non-empty one first and
  # json_str takes it off again: BSD/macOS sed ends its output with a newline even when the input has
  # none (GNU sed does not), which would corrupt probe names and error strings. The .cmd file already
  # ends with a newline after each argument.
  i=0
  while [ "$i" -lt "$PROBE_COUNT" ]; do
    base="$TMP_DIR/$i"
    for f in "$base.name" "$base.out.clean" "$base.err.clean" "$base.error"; do
      if [ -s "$f" ]; then
        printf '\n' >>"$f"
        files+=("$f")
      fi
    done
    files+=("$base.cmd.clean")
    i=$((i + 1))
  done
  if [ "${#files[@]}" -gt 0 ]; then
    sed -i.orig -f "$ESCAPE_SED" "${files[@]}"
  fi
  # platform, script name and the timestamp are fixed ASCII without quotes or backslashes.
  printf '{"schema": 1, "platform": "%s", "script": "%s", "collected_at": "%s", "probes": {' \
    "$PLATFORM" "$SCRIPT_NAME" "$collected_at"
  sep=''
  i=0
  while [ "$i" -lt "$PROBE_COUNT" ]; do
    base="$TMP_DIR/$i"
    json_str "$base.name"
    printf '%s\n %s: {"cmd": [' "$sep" "$REPLY"
    arg_sep=''
    while IFS= read -r line; do
      printf '%s"%s"' "$arg_sep" "$line"
      arg_sep=', '
    done <"$base.cmd.clean"
    IFS= read -r rc <"$base.rc" || true
    json_str "$base.out.clean"
    printf '], "exit_code": %s, "stdout": %s' "$rc" "$REPLY"
    json_str "$base.err.clean"
    printf ', "stderr": %s' "$REPLY"
    if [ -e "$base.error" ]; then
      json_str "$base.error"
      printf ', "error": %s' "$REPLY"
    fi
    printf '}'
    sep=','
    i=$((i + 1))
  done
  printf '\n}}\n'
}

pick_python() {
  if [ -n "${HC_PYTHON:-}" ]; then
    printf '%s' "$HC_PYTHON"
    return 0
  fi
  local py
  py=$(command -v python3 2>/dev/null) || return 1
  # macOS: /usr/bin/python3 is an installer stub until the Command Line Tools exist.
  if [ "$(uname -s)" = Darwin ] && [ "$py" = /usr/bin/python3 ] && ! xcode-select -p >/dev/null 2>&1; then
    return 1
  fi
  "$py" -c 'import json' >/dev/null 2>&1 || return 1
  printf '%s' "$py"
}

emit_json() {
  local collected_at mode py=''
  collected_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  mode=${HC_INVENTORY_JSON:-auto}
  case $mode in
    auto) py=$(pick_python) || py='' ;;
    python)
      py=$(pick_python) || { echo "HC_INVENTORY_JSON=python but no usable python3 was found" >&2; exit 2; }
      ;;
    bash) py='' ;;
    *) echo "HC_INVENTORY_JSON must be auto, python or bash (got '$mode')" >&2; exit 2 ;;
  esac
  write_utf8_sed
  local i=0
  while [ "$i" -lt "$PROBE_COUNT" ]; do
    clean_file "$TMP_DIR/$i.cmd" "$TMP_DIR/$i.cmd.clean"
    clean_file "$TMP_DIR/$i.out" "$TMP_DIR/$i.out.clean" redact
    clean_file "$TMP_DIR/$i.err" "$TMP_DIR/$i.err.clean" redact
    i=$((i + 1))
  done
  if [ -n "$py" ]; then
    "$py" -c "$PY_WRITER" "$TMP_DIR" "$PROBE_COUNT" "$PLATFORM" "$SCRIPT_NAME" "$collected_at"
  else
    write_json_bash "$collected_at"
  fi
}

# ------------------------------------------------------------------ Linux probes

NVCC=nvcc
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC=/usr/local/cuda/bin/nvcc
fi

probe uname_m uname -m
probe kernel uname -sr
probe os_release cat /etc/os-release
probe lscpu lscpu
probe free_b free -b
probe df_home df -B1 "${HOME:-/}"
probe ip_br_addr ip -br addr
probe nvidia_smi nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
probe nvcc_version "$NVCC" --version

emit_json
