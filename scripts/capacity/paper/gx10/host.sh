#!/bin/bash
# Start the Hypercompute host (gateway + dashboard) on the GX10 over Iroh, port 8001. The log holds a live invite: never print it whole.
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export HYPERCOMPUTE_LLAMA_BIN_DIR=$HOME/.hypercompute/src/llama.cpp/build-hypercompute/bin
cd ~/hypercompute
exec uv run --no-sync hypercompute host --transport iroh --port 8001 --no-browser
