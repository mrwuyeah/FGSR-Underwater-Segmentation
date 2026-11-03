#!/usr/bin/env bash
# Train a UniMatch-based model on SUIM.
#   usage: bash scripts/train_suim.sh CONFIG SPLIT EXP_NAME [NGPU] [PORT]
# Example:
#   bash scripts/train_suim.sh configs/suim_hybrid_fgsr.yaml 381 suim_full 4 29504
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 CONFIG SPLIT EXP_NAME [NGPU] [PORT]" >&2
    exit 1
fi

CONFIG="$1"
SPLIT="$2"
EXP="$3"
NGPU="${4:-4}"
PORT="${5:-29501}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

torchrun --nproc_per_node="$NGPU" --master_port="$PORT" unimatch.py \
    --config "$CONFIG" \
    --labeled-id-path "splits/suim/$SPLIT/labeled.txt" \
    --unlabeled-id-path "splits/suim/$SPLIT/unlabeled.txt" \
    --save-path "checkpoints/$EXP" \
    --port "$PORT"
