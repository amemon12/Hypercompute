#!/bin/bash
# Print each managed peer's host-side seed job and the peer's own seed report. No secrets.
cd /c/Users/operator/.hypercompute/worktrees/live-host && HYPERCOMPUTE_HOME='C:\Users\operator\.hypercompute\iroh-session' timeout 60 uv run --no-sync hypercompute capacity peers --port 8001 --json 2>&1 | /c/Users/operator/AppData/Local/Programs/Python/Python312/python.exe -I -c "
import json,sys
d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get('peers',d)
for p in d:
    r=p.get('seed_request') or {}
    rep=p.get('report') or {}
    s=rep.get('seed') or {}
    print(p.get('display_name') or p.get('name'), '| host', r.get('state'), r.get('detail'), 'tensors', r.get('tensors'),
          '| peer', s.get('state'), 'bytes', s.get('bytes_done'), '/', s.get('bytes_total'), 'present', s.get('present'), 'failed', s.get('failed'))
" 2>&1 | sed -E 's/[A-Za-z0-9_-]{40,}/<r>/g'
date -u +%H:%M:%S
