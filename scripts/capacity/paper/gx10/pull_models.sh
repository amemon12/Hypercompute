#!/bin/bash
# Pull the smoke model, the 120B demo model, then the 671B, one at a time; each is sha256-verified by hypercompute.
export PATH=$HOME/.local/bin:$PATH
cd ~/hypercompute
for m in qwen2.5-0.5b-instruct-q4_k_m gpt-oss-120b-mxfp4 deepseek-v3.1-ud-tq1_0; do
  echo "$(date -u +%H:%M:%S) pull $m"
  uv run --no-sync hypercompute models pull $m 2>&1 | awk "NR % 20 == 1 || /Verified|rror|already/"
  echo "$(date -u +%H:%M:%S) exit $? for $m"
done
