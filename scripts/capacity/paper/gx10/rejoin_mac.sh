#!/bin/bash
# rejoin_mac.sh NAME USER@HOST CAP RESERVE [EXTRA...]: mint a fresh invite on the GX10 host, then run mac_join.sh on the
# Mac over SSH. The invite only lives in shell variables; output is redacted. Never prints secrets.
set -uo pipefail
name="$1"; host="$2"; cap="$3"; res="$4"; shift 4; extra="$*"
K="-i $HOME/.ssh/hypercompute_ed25519 -o BatchMode=yes -o ConnectTimeout=15"
GX=asus@192.168.1.221
inv=$(timeout 60 ssh $K $GX 'cd ~/hypercompute && ~/.local/bin/uv run --no-sync hypercompute invite --port 8001' 2>&1) ||
  { echo "$name invite failed"; exit 1; }
line=$(grep -m1 -- "--ticket" <<<"$inv")
t=$(sed -E 's/.*--ticket ([^ ]+).*/\1/' <<<"$line"); c=$(sed -E 's/.*--code ([^ ]+).*/\1/' <<<"$line")
unset inv line
[ -n "$t" ] && [ -n "$c" ] || { echo "$name could not parse invite"; exit 1; }
timeout 90 ssh $K "$host" "${MACENV:-} ~/mac_join.sh '$t' '$c' '$name' '$cap' '$res' $extra" 2>&1 |
  sed -E 's/[A-Za-z0-9_-]{30,}/<r>/g; s/[A-Z0-9]{5}-[A-Z0-9]{5}/<code>/g'
