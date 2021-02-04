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
    'experimental/users/xiuming/sim/brdf/microfacet:render' \
    -- \
    --default_rough='0.1' \
    --envmap_path='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/envmaps/blender/interior.exr' \
    --envmap_h='64' \
    --envmap_inten='0.5' \
    --ims='256' \
    --spp='1' \
    --out_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/tmp/microfacet-render' \
    --debug='False' \
    "$@"
exit

blaze run -c opt --copt=-mavx \
    'experimental/users/xiuming/sim/brdf/microfacet:render' \
    -- \
    --default_rough='0.3' \
    --envmap_path='point' \
    --envmap_h='16' \
    --envmap_inten='2.' \
    --ims='256' \
    --spp='1' \
    --out_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/tmp/microfacet-render' \
    --debug='False' \
    "$@"
exit
