#!/usr/bin/env bash
# Train one arm (or all five, one after another) with the released recipe.
#
#   bash train.sh sage              # one arm: happo_tree | happo_tree_cbf | sail | talo | sage
#   bash train.sh all               # the five arms in that order
#   bash train.sh sage --foo bar    # extra train_sage.py flags are appended (changes the recipe)
#   SCENE=<name> bash train.sh sage # scene file isaac_sage/scenes/<name>.json
#   DRY_RUN=1 bash train.sh all     # print the commands only
#
# Each run writes results/<scene>_<arm>_seed<seed>_<date>/ (run_args.json, train.log,
# tensorboard/, models/) and is restarted from its latest checkpoint if the
# process dies before the run is finished (isaac_sage/run_resumable.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ARMS=(happo_tree happo_tree_cbf sail talo sage)
usage() { echo "usage: bash train.sh {happo_tree|happo_tree_cbf|sail|talo|sage|all} [extra train_sage.py flags]" >&2; exit 1; }
[ $# -ge 1 ] || usage
ARM="$1"; shift

export ISAACLAB_CORE_ROOT="${ISAACLAB_CORE_ROOT:-$ROOT/third_party/IsaacLab}"
[ -d "$ISAACLAB_CORE_ROOT/source/isaaclab/isaaclab" ] || {
    echo "Isaac Lab checkout not found at $ISAACLAB_CORE_ROOT (run install.sh)"; exit 1; }
export PYTHON="${PYTHON:-python}"
SEED="${SEED:-1}"
SCENE="${SCENE:-tilt_gate_2m2}"
SCENE_JSON="isaac_sage/scenes/${SCENE}.json"
[ -f "$SCENE_JSON" ] || { echo "scene file not found: $SCENE_JSON" >&2; exit 1; }

# 128 envs x 60 high-level steps = 7680 env-steps per iteration.  The tau / kappa /
# curriculum schedules are planned over 2,500,000 env-steps and training stops
# after 200 iterations.
NUM_ENVS=128
SCHEDULE_ENV_STEPS=2500000
ITERATIONS=200

COMMON=(
    --seed "$SEED"
    --num_envs "$NUM_ENVS" --num_steps 60 --num_env_steps "$SCHEDULE_ENV_STEPS" --max_iters "$ITERATIONS"
    --scene_json "$SCENE_JSON"
    --resume isaac_sage/assets/init/phase1_init.pt --activate_lidar
    --tau_init 2.0 --tau_final 0.4 --tau_anneal_start 0.112 --tau_anneal_end 0.36
    --hard_finetune_fraction 1.0
    --kappa_init 8 --kappa_final 32
    --w_heading 0 --w_obstacle 0 --w_collision_contact 0.2
    --start_curriculum_prob 0.7
    --projection_progress --success_mode loose --partner_noise --success_compensation
    --actor_lr 1e-4 --critic_lr 5e-5
    --obstacle_margin 0.1
    --headless
)

arm_args() {
    case "$1" in
        happo_tree|happo_tree_cbf) echo "--actor_update happo" ;;            # joint-observation critic (auto)
        sail)          echo "--actor_update happo --critic joint" ;;
        talo|sage)     echo "" ;;                                # g_TALO, team-context critic (auto)
        *) usage ;;
    esac
}

run_arm() {
    local arm="$1"; shift
    local log_dir="results/${SCENE}_${arm}_seed${SEED}_$(date +%Y-%m-%d-%H-%M-%S)"
    # shellcheck disable=SC2046
    local cmd=(--variant "$arm" "${COMMON[@]}" $(arm_args "$arm") "$@")
    echo "[train.sh] $arm -> $log_dir"
    echo "[train.sh] python -m isaac_sage.train_sage ${cmd[*]}"
    if [ "${DRY_RUN:-0}" = "1" ]; then return; fi
    bash isaac_sage/run_resumable.sh "$log_dir" "${cmd[@]}"
    echo "[train.sh] $arm finished: $log_dir"
}

if [ "$ARM" = "all" ]; then
    for a in "${ARMS[@]}"; do run_arm "$a" "$@"; done
else
    arm_args "$ARM" >/dev/null
    run_arm "$ARM" "$@"
fi
