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


blaze run -c opt --copt=-mavx \
    'experimental/users/xiuming/sim/brdf/merl:render' \
    -- \
    --merl_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/brdf/merl/' \
    --envmap_path='point' \
    --envmap_h='16' \
    --envmap_inten='40' \
    --slice_percentile='80' \
    --ims='512' \
    --spp='1' \
    --out_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/brdf/merl_render' \
    --debug='False' \
    --lambert_override='False' \
    --disney_paper_subset='False' \
    "$@"
exit
