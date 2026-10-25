#!/bin/bash
# join_g14.sh: join this G14 to the GX10 host as a CUDA peer (default home, pinned windows-x64-cuda13.4 engine).
# Runs in the foreground; output is redacted. The invite only lives in shell variables. Never prints secrets.
set -uo pipefail
K="-i $HOME/.ssh/hypercompute_ed25519 -o BatchMode=yes -o ConnectTimeout=15"
inv=$(timeout 60 ssh $K asus@192.168.1.221 'cd ~/hypercompute && ~/.local/bin/uv run --no-sync hypercompute invite --port 8001' 2>&1) ||
  { echo "invite failed"; exit 1; }
line=$(grep -m1 -- "--ticket" <<<"$inv")
t=$(sed -E 's/.*--ticket ([^ ]+).*/\1/' <<<"$line"); c=$(sed -E 's/.*--code ([^ ]+).*/\1/' <<<"$line")
unset inv line
[ -n "$t" ] && [ -n "$c" ] || { echo "could not parse invite"; exit 1; }
unset HYPERCOMPUTE_HOME HYPERCOMPUTE_CONFIG_DIR HYPERCOMPUTE_LLAMA_BIN_DIR
cd /c/Users/operator/.hypercompute/worktrees/live-host
/c/Users/operator/.local/bin/uv.exe run --no-sync hypercompute join --ticket "$t" --code "$c" --name g14 \
  --devices CUDA0 --max-memory "${G14_MAX_MEMORY:-9GiB}" --yes --cache --managed --devices-allowed CUDA0 \
  --max-disk "${G14_MAX_DISK:-24GiB}" 2>&1 | sed -u -E 's/[A-Za-z0-9_-]{30,}/<r>/g; s/[A-Z0-9]{5}-[A-Z0-9]{5}/<code>/g'
