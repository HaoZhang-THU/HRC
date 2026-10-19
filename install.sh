#!/usr/bin/env bash
# One-time setup: conda env with the pinned python packages (Isaac Sim 4.5.0,
# torch 2.5.1+cu124, ...), the pinned Isaac Lab commit in third_party/IsaacLab,
# this package in editable mode, then the CPU smoke test.
#
#   bash install.sh                 # env name: isaac_sage
#   ENV_NAME=myenv bash install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${ENV_NAME:-isaac_sage}"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="3e73d6dd79080fd7632488c061052a6edd52e230"
ISAACLAB_DIR="$ROOT/third_party/IsaacLab"

# third_party/IsaacLab must be the real clone made below.  A symlink (e.g. to another
# Isaac Lab checkout, or one that came along in an archive and is broken here) would
# make this script check out the pinned commit in someone else's repository, or fail
# to clone; run against another checkout with ISAACLAB_CORE_ROOT instead.
if [ -L "$ISAACLAB_DIR" ]; then
    if [ -e "$ISAACLAB_DIR" ]; then
        echo "error: $ISAACLAB_DIR is a symlink to $(readlink "$ISAACLAB_DIR")." >&2
    else
        echo "error: $ISAACLAB_DIR is a broken symlink (-> $(readlink "$ISAACLAB_DIR"))," >&2
        echo "       probably copied along with the release archive." >&2
    fi
    echo "       Remove it (rm \"$ISAACLAB_DIR\") and rerun install.sh, which clones Isaac Lab there." >&2
    echo "       To use an existing checkout of commit $ISAACLAB_COMMIT instead, skip install.sh's clone" >&2
    echo "       and run train.sh / eval.sh with ISAACLAB_CORE_ROOT=/path/to/that/checkout." >&2
    exit 1
fi
if [ -e "$ISAACLAB_DIR" ] && [ ! -d "$ISAACLAB_DIR/.git" ]; then
    echo "error: $ISAACLAB_DIR exists but is not a git clone of Isaac Lab; remove it and rerun install.sh." >&2
    exit 1
fi

command -v conda >/dev/null || { echo "conda not found (install Miniconda / Miniforge first)"; exit 1; }
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

python -m pip install \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r "$ROOT/requirements.lock.txt"

if [ ! -d "$ISAACLAB_DIR/.git" ]; then
    mkdir -p "$(dirname "$ISAACLAB_DIR")"
    git clone "$ISAACLAB_REPO" "$ISAACLAB_DIR"
fi
git -C "$ISAACLAB_DIR" fetch --quiet origin "$ISAACLAB_COMMIT" || true
git -C "$ISAACLAB_DIR" checkout --quiet "$ISAACLAB_COMMIT"
echo "Isaac Lab at $(git -C "$ISAACLAB_DIR" rev-parse HEAD)"

python -m pip install --no-deps -e "$ROOT"

cd "$ROOT"
python -m isaac_sage.smoke_test

cat <<EOF

Setup done.  Before the first run, read the NVIDIA Omniverse EULA and accept it with
    export OMNI_KIT_ACCEPT_EULA=YES
Then train with
    conda activate $ENV_NAME
    bash train.sh sage          # or tree / tree_cbf / sail / talo / all
EOF
