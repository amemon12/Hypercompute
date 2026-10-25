#!/bin/bash
HC_HERE="$(cd "$(dirname "$0")" && pwd)" || exit 1
. "$HC_HERE/lib/hc-common.sh"
hc_main invite "$@"
