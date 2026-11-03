#!/usr/bin/env bash
# Train the U2PL cross-framework port on SUIM.
#   usage: bash scripts/train_u2pl_suim.sh CONFIG [NGPU] [PORT]
# Example:
#   bash scripts/train_u2pl_suim.sh u2pl/configs/suim_1_4_fgsr.yaml 4 29509
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 CONFIG [NGPU] [PORT]" >&2
    exit 1
fi

CONFIG="$1"
NGPU="${2:-4}"
PORT="${3:-29508}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Make the config path absolute before changing the working directory.
if [[ "$CONFIG" = /* ]]; then
    CONFIG_ABS="$CONFIG"
else
    CONFIG_ABS="$ROOT/$CONFIG"
fi

cd "$ROOT/u2pl"

torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train_semi.py \
    --config "$CONFIG_ABS" \
    --port "$PORT"
