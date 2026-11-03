#!/usr/bin/env bash
# Train a UniMatch-based model on DUT-USEG (1/4 split: 345 labeled images).
#   usage: bash scripts/train_dut.sh CONFIG EXP_NAME [NGPU] [PORT]
# Example:
#   bash scripts/train_dut.sh configs/dut_hybrid_fgsr.yaml dut_full 4 29506
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 CONFIG EXP_NAME [NGPU] [PORT]" >&2
    exit 1
fi

CONFIG="$1"
EXP="$2"
NGPU="${3:-4}"
PORT="${4:-29505}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

torchrun --nproc_per_node="$NGPU" --master_port="$PORT" unimatch.py \
    --config "$CONFIG" \
    --labeled-id-path splits/dut/345/labeled.txt \
    --unlabeled-id-path splits/dut/345/unlabeled.txt \
    --save-path "checkpoints/$EXP" \
    --port "$PORT"
