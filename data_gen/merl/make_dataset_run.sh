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


ims='512'
envmap_h='16'
spp='1'

blaze run -c opt --copt=-mavx \
    'experimental/users/xiuming/sim/data_gen/merl:make_dataset' \
    -- \
    --merl_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/brdf/merl/' \
    --envmap_h="$envmap_h" \
    --ims="$ims" \
    --spp="$spp" \
    --out_dir="/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/brdf/merl_npz/ims${ims}_envmaph${envmap_h}_spp${spp}" \
    "$@"
exit
