#!/bin/bash
# After the 671B download: verify, register, plan layout C, seed every Mac from Hugging Face by that plan, wait for the
# host's hashing to finish, then run the G14-alone baseline. Prints one line per step; never prints secrets.
set -uo pipefail
S=/c/Users/operator/AppData/Local/Temp/claude/C--/9e2c80d5-50ca-460d-8443-8db45359fcde/scratchpad
P=/c/Users/operator/AppData/Local/Programs/Python/Python312/python.exe
LIVE=/c/Users/operator/.hypercompute/worktrees/live-host
M=deepseek-v3.1-ud-tq1_0
NODES="4c2991743c154be896d10a0b53ba4d2b 6dad6cf75b2e4cf197b7bc7f4d0a67e4 d4e65f83c4be4a41aad722e40983b08b"
export HYPERCOMPUTE_HOME='C:\Users\operator\.hypercompute\iroh-session'
export HYPERCOMPUTE_CONFIG_DIR='C:\Users\operator\.hypercompute\worktrees\live-host\configs'
step() { echo "$(date -u +%H:%M:%S) $*"; }

for _ in $(seq 1 1440); do grep -qE "sha256 (OK|MISMATCH)|^incomplete|giving up" $S/ds_download.log && break; sleep 15; done
grep -q "sha256 OK" $S/ds_download.log || { step "download did not verify: $(tail -1 $S/ds_download.log)"; exit 1; }
step "download verified: $(grep 'sha256' $S/ds_download.log)"

cd $LIVE
timeout 120 uv run --no-sync python -c "
from hypercompute.services.models import ModelService
s = ModelService(); s._write_marker('$M'); print('verified marker', s._verified('$M'))
" || { step "marker failed"; exit 1; }

args=""; for n in $NODES; do args="$args --node $n"; done
timeout 300 uv run --no-sync python scripts/capacity/run_big.py --model $M --plan-only --layout "$S/runs/layout_671b_C.json" \
  $args --min-available-gib 0 >/dev/null 2>&1
d=$(ls -td results/capacity/*big-$M* | head -1)
plan=$($P -I -c "import json,sys; d=json.load(open(sys.argv[1]+'/run.json')); p=(d.get('predicted') or {}).get('plan') or {}; print(p.get('id') or '')" "$d")
status=$($P -I -c "import json,sys; d=json.load(open(sys.argv[1]+'/run.json')); e=d.get('error'); print(d['status'], (e or {}).get('message','') if isinstance(e,dict) else e or '')" "$d")
step "layout C plan: $status ${plan:0:8}"
[ -n "$plan" ] || exit 1
for n in $NODES; do
  timeout 120 uv run --no-sync hypercompute capacity seed --node ${n:0:8} --model $M --plan $plan --port 8001 2>&1 \
    | sed -E 's/[A-Za-z0-9_-]{40,}/<r>/g' | tail -1
done
for _ in $(seq 1 240); do o=$(bash $S/peers.sh); echo "$o" | grep -q "host hashing" || break; sleep 15; done
step "host hashing done"; bash $S/peers.sh

$P -I -c "
import json
S='C:/Users/operator/AppData/Local/Temp/claude/C--/9e2c80d5-50ca-460d-8443-8db45359fcde/scratchpad/runs/'
json.dump([{'label':'671b-local-1','model':'$M','nodes':[],'args':['--prompt-tokens','64','--output-tokens','32','--warmup','1','--reps','2','--load-timeout','3600'],'timeout_s':7200}], open(S+'plan_671b_A.json','w'), indent=1)"
step "baseline run starting"
$P -I $S/matrix.py $S/runs/plan_671b_A.json $S/runs/index_671b.jsonl
cp "$LIVE/scripts/capacity/batchlib.py" $S/  # summarize.py imports it from beside itself in a standalone copy
$P -I $S/summarize.py $S/runs/summary_671b.md $S/runs/index_671b.jsonl | grep '^| 671b-local-1 ' | head -1
step "seed state after baseline"; bash $S/peers.sh
