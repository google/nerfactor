#!/usr/bin/env bash

set -e

PYTHONPATH='/data/vision/billf/shapetime/new1/xiuminglib:/data/vision/billf/intrinsic/sim/code/nerfactor' \
    python '/data/vision/billf/intrinsic/sim/code/nerfactor/data_gen/dtu_mvs/surf_from_mvs.py' \
    --cam_dir='/data/vision/billf/intrinsic/sim/data/dtu-mvs/SampleSet/MVS_Data/Calibration/cal18' \
    --surf_dir='/data/vision/billf/intrinsic/sim/data/dtu-mvs/Surfaces/furu' \
    --img_root='/data/vision/billf/intrinsic/sim/data/dtu-mvs/Rectified' \
    --img_outroot='/data/vision/billf/intrinsic/sim/data/dtu-mvs_proc-from-raw' \
    --surf_outroot='/data/vision/billf/intrinsic/sim/output/surf_mvs' \
    --scenes='scan105,scan106,scan110,scan114,scan118,scan122,scan24,scan37,scan40,scan55,scan63,scan65,scan69,scan83,scan97' \
    "$@"
