#!/bin/bash
# Serve a GGUF on the GB10 with every layer on the GPU and report prefill/decode tok/s for N runs.
# Usage: smoke.sh MODEL.gguf [N_PREDICT] [RUNS] [extra llama-server args...]
B=$HOME/.hypercompute/src/llama.cpp/build-hypercompute/bin
M=$1; NP=${2:-128}; RUNS=${3:-2}; shift 3 2>/dev/null || shift $#
$B/llama-server -m "$M" -ngl 99 --host 127.0.0.1 --port 8090 -c 4096 "$@" > /tmp/smoke.log 2>&1 &
P=$!
for i in $(seq 1 900); do curl -sf 127.0.0.1:8090/health >/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 1; done
echo "ready after ${i}s"
for r in $(seq 1 $RUNS); do
  curl -s 127.0.0.1:8090/completion -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Explain what a GPU does in three sentences.\",\"n_predict\":$NP,\"temperature\":0,\"cache_prompt\":false}" |
    python3 -c "import json,sys
d=json.load(sys.stdin); t=d[\"timings\"]
print(\"run prefill %.1f tok/s (%d tok), decode %.2f tok/s (%d tok)\" % (t[\"prompt_per_second\"], t[\"prompt_n\"], t[\"predicted_per_second\"], t[\"predicted_n\"]))
print(\"   \", repr(d[\"content\"][:140]))"
done
kill $P; wait $P 2>/dev/null
grep -E "offloaded .* layers|model buffer size" /tmp/smoke.log | head -4
