#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 scene[ ...]"
    exit 1
fi
scene="$1"
shift # shift the remaining arguments

PYTHONPATH='/data/vision/billf/shapetime/new1/xiuminglib:/data/vision/billf/intrinsic/sim/code/nerfactor' \
    python '/data/vision/billf/intrinsic/sim/code/nerfactor/data_gen/dtu_mvs/make_dataset.py' \
    --scene_dir="/data/vision/billf/intrinsic/sim/data/dtu-mvs/dtu_dataset/rs_dtu_4/DTU/$scene" \
    --outroot="/data/vision/billf/intrinsic/sim/data/dtu-mvs_proc/$scene" \
    "$@"
