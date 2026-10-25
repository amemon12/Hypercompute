#!/bin/bash
# rejoin.sh NAME USER@HOST CAP RESERVE : fresh invite from the gateway, then mac_join.sh over SSH. Never prints secrets.
# MACENV (optional) prefixes the remote command, e.g. MACENV=HYPERCOMPUTE_STOCK_BRIDGE=1 for the bridge A/B.
set -uo pipefail
S=/c/Users/operator/AppData/Local/Temp/claude/C--/9e2c80d5-50ca-460d-8443-8db45359fcde/scratchpad
name="$1"; host="$2"; cap="$3"; res="$4"; shift 4; extra="$*"
K="-i $HOME/.ssh/hypercompute_ed25519 -o BatchMode=yes -o ConnectTimeout=15"
f="$S/inv_$name.txt"
( cd /c/Users/operator/.hypercompute/worktrees/live-host && HYPERCOMPUTE_HOME='C:\Users\operator\.hypercompute\iroh-session' timeout 60 uv run --no-sync hypercompute invite --port 8001 > "$f" 2>&1 ) || { echo "$name invite failed"; exit 1; }
line=$(grep -m1 -- "--ticket" "$f")
t=$(sed -E 's/.*--ticket ([^ ]+).*/\1/' <<<"$line"); c=$(sed -E 's/.*--code ([^ ]+).*/\1/' <<<"$line")
[ -n "$t" ] && [ -n "$c" ] || { echo "$name could not parse invite"; exit 1; }
timeout 90 ssh $K "$host" "${MACENV:-} ~/mac_join.sh '$t' '$c' '$name' '$cap' '$res' $extra" 2>&1 | sed -E 's/[A-Za-z0-9_-]{30,}/<r>/g'
