#!/usr/bin/env bash

set -e

# scan105,scan106,scan110,scan114,scan118,scan122,scan24,scan37,scan40,scan55,
# scan63,scan65,scan69,scan83,scan97
if [ $# -lt 1 ]; then
    echo "Usage: $0 scene[ ...]"
    exit 1
fi
scene="$1"
shift # shift the remaining arguments

PYTHONPATH='/data/vision/billf/shapetime/new1/xiuminglib:/data/vision/billf/intrinsic/sim/code/nerfactor' \
    python '/data/vision/billf/intrinsic/sim/code/nerfactor/data_gen/dtu_mvs/surf_from_mvs.py' \
    --cam_dir='/data/vision/billf/intrinsic/sim/data/dtu-mvs/SampleSet/MVS_Data/Calibration/cal18' \
    --surf_dir='/data/vision/billf/intrinsic/sim/data/dtu-mvs/Surfaces/furu' \
    --img_dir="/data/vision/billf/intrinsic/sim/data/dtu-mvs/Rectified/$scene" \
    --outdir="/data/vision/billf/intrinsic/sim/output/surf_mvs/$scene" \
    "$@"
