#!/bin/bash
# Run on a Mac over SSH as the hypercompute user: join the G14 host in the background, kept awake by caffeinate,
# surviving the SSH session ending. Usage: mac_join.sh TICKET CODE NAME [MAX_MEMORY] [RESERVE] [EXTRA FLAGS...]
# EXTRA FLAGS are passed through to `hypercompute join`, e.g. --managed --max-disk 100GiB.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
TICKET="$1"; CODE="$2"; NAME="$3"; MAXMEM="${4:-10GiB}"; RESERVE="${5:-4GiB}"
shift $(( $# < 5 ? $# : 5 ))
cd "${HC_REPO:-$HOME/hc-peer}"
uv run hypercompute leave >/dev/null 2>&1 || true
pkill -f "hypercompute join" >/dev/null 2>&1 || true
nohup caffeinate -dimsu uv run hypercompute join --ticket "$TICKET" --code "$CODE" --name "$NAME" \
  --max-memory "$MAXMEM" --reserve "$RESERVE" --devices MTL0 --cache --yes "$@" > "$HOME/hc-join.log" 2>&1 &
echo "started pid $!"
sleep 8
tail -5 "$HOME/hc-join.log"
