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
    'experimental/users/xiuming/sim/sim/playground/compare_nerf_quality:vs_jaxnerf' \
    -- \
    --my_result_root='/output/train/nerf_lego.ini_render_lego_studio_512_accu-sigma/lr0.001_mgm-1/vis_vali/epoch000004700/' \
    --jax_result_root='/nerf/blender_09220103/17459230_1/test_preds/' \
    --out_dir='/usr/local/home/xiuming/Desktop/compare_nerf_quality' \
    "$@"
