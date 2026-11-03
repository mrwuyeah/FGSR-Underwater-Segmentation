#!/usr/bin/env bash
# Fully supervised reference on all 1,525 SUIM training images.
#   usage: bash scripts/train_suim_supervised.sh CONFIG [NGPU] [PORT]
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 CONFIG [NGPU] [PORT]" >&2
    exit 1
fi

CONFIG="$1"
NGPU="${2:-4}"
PORT="${3:-29507}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

torchrun --nproc_per_node="$NGPU" --master_port="$PORT" supervised.py \
    --config "$CONFIG" \
    --labeled-id-path splits/suim/full/labeled.txt \
    --save-path checkpoints/suim_supervised_full \
    --port "$PORT"
