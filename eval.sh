#!/usr/bin/env bash
# Deployment evaluation (Algorithm 2) of a run produced by train.sh, on the run's
# training scene.  The scene and the arm's env toggles are read from the run's
# run_args.json.
#
#   bash eval.sh results/<scene>_sage_seed1_<date>                 # models/iter_200.pt, 30 episodes
#   bash eval.sh results/<scene>_sage_seed1_<date> models/latest.pt
#   bash eval.sh <run_dir> <ckpt> --shield off             # extra eval_sage.py flags
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[ $# -ge 1 ] || { echo "usage: bash eval.sh <run_dir> [checkpoint relative to run_dir] [extra eval_sage.py flags]" >&2; exit 1; }
RUN_DIR="$1"; shift
CKPT="models/iter_200.pt"
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CKPT="$1"; shift; fi
[ -f "$RUN_DIR/$CKPT" ] || { echo "checkpoint not found: $RUN_DIR/$CKPT" >&2; exit 1; }

export ISAACLAB_CORE_ROOT="${ISAACLAB_CORE_ROOT:-$ROOT/third_party/IsaacLab}"
PYTHON="${PYTHON:-python}"
SCENE_JSON="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["scene_json"])' "$RUN_DIR/run_args.json")"

"$PYTHON" -m isaac_sage.eval_sage \
    --checkpoint "$RUN_DIR/$CKPT" \
    --run_args "$RUN_DIR/run_args.json" \
    --scene_json "$SCENE_JSON" \
    --num_episodes 30 \
    --out "$RUN_DIR/eval_$(basename "$CKPT" .pt).json" \
    --headless "$@"
