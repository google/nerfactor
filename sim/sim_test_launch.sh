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

cd "$(p4 g4d)"

#: <<'END'
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
#END

# Actual inference
target='experimental/users/xiuming/sim/sim/sim_test.par'
rabbit --verifiable build -c opt "$target"
readonly G3BIN="$(rabbit info blaze-bin -c opt --force_python=PY3)"
"${G3BIN}/${target}" \
    --ckpt="$latest_ckpt" \
    --mode='launch' \
    --color_correct_albedo \
    --gfs_user='gcam-gpu' \
    --flume_borg_user_name='gcam-gpu' \
    --flume_borg_accounting_charged_user_name='gcam-gpu' \
    --flume_borg_cells='is' \
    --flume_worker_priority='200' \
    --flume_use_batch_scheduler='false' \
    --flume_batch_scheduler_strategy='RUN_WITHIN' \
    --flume_batch_scheduler_start_deadline_secs='60' \
    --flume_clean_up_tmp_dirs='ALWAYS' \
    --flume_worker_remote_hdd_scratch='512M' \
    --flume_dax_num_threads_per_worker='1' \
    --alsologtostderr \
    "$@"

cd -
