#!/bin/bash
# hcpeer S1 gate, run on a Mac: one greedy completion through the stock rpc-server (CPU device) and twice through
# the hcpeer clone with HC_PEER_MMAP=1 (first load saves the cache, second load maps it). All three texts must match.
set -euo pipefail
# Full Xcode refuses python3 and git until its license is accepted (sudo); the standalone tools do not.
if [ -d /Library/Developer/CommandLineTools ]; then export DEVELOPER_DIR=/Library/Developer/CommandLineTools; fi
E="$HOME/.hypercompute/engines/llama-b11770-macos-arm64/llama-b11770"
EH="$E-hcpeer"
W="$HOME/hcpeer-try/gate"
case "${1:-0.5b}" in
  0.5b) FILES="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db" ;;
  7b) FILES="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf 539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a" ;;
  *) echo "usage: hcpeer_gate.sh [0.5b|7b]"; exit 2 ;;
esac
RPC_PORT=47170
SRV_PORT=47180
mkdir -p "$W/models"
rm -rf "$W/cache"
M=''
set -- $FILES
while [ $# -ge 2 ]; do
  f="$W/models/$(basename "$1")"
  [ -n "$M" ] || M="$f"
  if [ ! -f "$f" ] || [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" != "$2" ]; then
    curl -sfL -o "$f.part" "$1" && mv "$f.part" "$f"
  fi
  [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$2" ] || { echo "sha256 mismatch for $f"; exit 1; }
  shift 2
done

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

wait_http() {
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$SRV_PORT/health" >/dev/null 2>&1 && return 0; sleep 0.5; done
  return 1
}

complete() {  # $1 label: start llama-server against the running rpc-server, one greedy completion, stop it
  "$E/llama-server" -m "$M" --rpc "127.0.0.1:$RPC_PORT" --device RPC0 -ngl 99 -c 1024 --host 127.0.0.1 \
    --port "$SRV_PORT" >"$W/server-$1.log" 2>&1 &
  local sp=$!
  wait_http || { echo "llama-server did not start ($1)"; tail -5 "$W/server-$1.log"; kill $sp; return 1; }
  curl -sf "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The three laws of thermodynamics are","n_predict":64,"temperature":0,"seed":7,"cache_prompt":false}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])' >"$W/out-$1.txt"
  kill $sp; wait $sp 2>/dev/null || true
}

start_rpc() {  # $1 engine dir, $2 log, then env assignments are inherited from the caller
  "$1/ggml-rpc-server" -H 127.0.0.1 -p "$RPC_PORT" -d CPU "${@:3}" >"$2" 2>&1 &
  PIDS+=($!)
  # Metal kernels compile at startup even with -d CPU; wait for the port, not a fixed delay.
  for _ in $(seq 1 120); do nc -z 127.0.0.1 "$RPC_PORT" 2>/dev/null && return 0; sleep 0.5; done
  echo "rpc-server did not listen on $RPC_PORT"; tail -5 "$2"; return 1
}

start_rpc "$E" "$W/rpc-stock.log"
complete stock
kill "${PIDS[${#PIDS[@]}-1]}"; sleep 1

HC_PEER_MMAP=1 HC_PEER_CACHE_MIN_FREE_GIB=0 LLAMA_CACHE="$W/cache" start_rpc "$EH" "$W/rpc-hcpeer.log" -c
complete hcpeer-save
complete hcpeer-map

for f in stock hcpeer-save hcpeer-map; do printf '%s: ' "$f"; head -c 120 "$W/out-$f.txt" | tr '\n' ' '; echo; done
if cmp -s "$W/out-stock.txt" "$W/out-hcpeer-save.txt" && cmp -s "$W/out-stock.txt" "$W/out-hcpeer-map.txt"; then
  echo "GATE PASS: identical greedy text through stock, hcpeer first load and hcpeer mapped load"
else
  echo "GATE FAIL: outputs differ"
fi
echo "cache files: $(ls "$W/cache/rpc" 2>/dev/null | wc -l | tr -d ' ')"
grep -iE "hcpeer" "$W/rpc-hcpeer.log" | sort | uniq -c | sort -rn | head -8
