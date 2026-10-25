#!/bin/bash
# matrix.sh PLAN INDEX: run the paper driver on the GX10 host.
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export HYPERCOMPUTE_LLAMA_BIN_DIR=$HOME/.hypercompute/src/llama.cpp/build-hypercompute/bin
export HC_PAPER_REPO=$HOME/hypercompute HC_PAPER_HOME=$HOME/.hypercompute HC_PAPER_UV=$HOME/.local/bin/uv HC_PAPER_PROBE_MACS=0
cd ~/hypercompute
exec python3 -I scripts/capacity/paper/matrix.py "$1" "$2"
