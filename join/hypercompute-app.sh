#!/bin/bash
# Double-click to open the Hypercompute app (host a pool or join one) in your browser.
cd "$(dirname "$0")/.." || exit 1
exec uv run --no-sync python app/hcapp.py "$@"
