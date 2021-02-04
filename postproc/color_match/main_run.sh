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

blaze run -c opt --copt=-mavx \
    'experimental/users/xiuming/sim/postproc/color_match:main' \
    -- \
    --data_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/ficus_2188/' \
    --data_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/hotdog_3072_no-ambient/' \
    --data_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/drums_2188/' \
    --light_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/envmaps/for-render_h16/' \
    --run_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_2188/lr0.001/' \
    --run_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/' \
    --run_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188/lr0.001/' \
    --out_root='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/results_s2021/color-matched' \
    --gfs_user='gcam-gpu' \
    "$@"
    #--data_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/hotdog_2163/' \
    #--run_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_2163_tv-5e-6/lr0.005/' \
exit
