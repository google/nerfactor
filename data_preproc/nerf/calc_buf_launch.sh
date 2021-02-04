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

id="$1"
spp='1'
light_h='16'

root='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim'

cd "$(p4 g4d)"

# Grab the latest checkpoint independently in case the model is training
t=$(date +"%s")
latest_ckpt_tmp="/tmp/latest_ckpt_$t"
blaze run -c opt \
    'experimental/users/xiuming/sim/sim:grab_latest_ckpt' \
    -- \
    --id="nerf_$id" \
    --out_file="$latest_ckpt_tmp" \
    --gfs_user='gcam-gpu'
latest_ckpt=$(cat "$latest_ckpt_tmp")

# Actual inference
target='experimental/users/xiuming/sim/data_preproc/nerf/calc_buf.par'
rabbit --verifiable build -c opt "$target"
readonly G3BIN="$(rabbit info blaze-bin -c opt --force_python=PY3)"
"${G3BIN}/${target}" \
    --ckpt="$latest_ckpt" \
    --data_root="$root/data/render_s2021/$id" \
    --spp="$spp" \
    --light_h="$light_h" \
    --out_root="$root/output/surf_s2021/${id}_lh${light_h}_spp${spp}" \
    --mode='flume' \
    --gfs_user='gcam-gpu' \
    --flume_borg_user_name='gcam-gpu' \
    --flume_borg_accounting_charged_user_name='gcam-gpu' \
    --flume_borg_cells='is' \
    --flume_use_batch_scheduler='false' \
    --flume_worker_priority='200' \
    --flume_worker_ram='64G' \
    --flume_worker_remote_hdd_scratch='512M' \
    --flume_batch_scheduler_strategy='RUN_WITHIN' \
    --flume_batch_scheduler_start_deadline_secs='60' \
    --flume_clean_up_tmp_dirs='ALWAYS' \
    --flume_dax_num_threads_per_worker='1' \
    --alsologtostderr \
    "$@"

cd -
