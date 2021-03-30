#!/usr/bin/env bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

id='ours_hotdog_3072'

cd "$(p4 g4d)"

: <<'END'
# Grab the latest checkpoint independently in case the model is training
t=$(date +"%s")
latest_ckpt_tmp="/tmp/latest_ckpt_$t"
blaze run -c opt \
    'experimental/users/xiuming/sim/sim:grab_latest_ckpt' \
    -- \
    --id="$id" \
    --out_file="$latest_ckpt_tmp" \
    --gfs_user='gcam-gpu'
latest_ckpt=$(cat "$latest_ckpt_tmp")
END
latest_ckpt='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/vis_test/ckpt-9'

# Actual inference
blaze run -c opt \
    'experimental/users/xiuming/sim/sim:sim_test' \
    -- \
    --ckpt="$latest_ckpt" \
    --mode='local' \
    --color_correct_albedo \
    --gfs_user='gcam-gpu' \
    "$@"
    # --debug

cd -
