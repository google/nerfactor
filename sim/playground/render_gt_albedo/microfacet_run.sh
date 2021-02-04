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


export TF_FORCE_GPU_ALLOW_GROWTH=true

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim/playground/render_gt_albedo:microfacet' \
    -- \
    --ckpt='/output/train/albedo_lego.ini_render_lego_studio_512/lr0.0001_mgm-1/checkpoints/ckpt-337' \
    --data_mode='train' \
    --nerf_data_dir='/tmp/calc_buf/train_033' \
    --albedo_path='/data/render/lego_studio_512/train_033/albedo.png' \
    --out_dir='/usr/local/home/xiuming/Desktop/render_gt_albedo' \
    "$@"
