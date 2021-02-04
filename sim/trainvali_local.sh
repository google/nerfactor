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
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_pixlight.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_microfacet_pixlight.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='nerf.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='light_findz.ini' \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_gtlight.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_microfacet.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_microfacet_gtlight.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='ns_shape.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='light.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='brdf.ini' \
    --debug \
    "$@"
exit

blaze run -c opt --copt=-mavx --config=cuda \
    'experimental/users/xiuming/sim/sim:trainvali' \
    -- \
    --config='nerf_sfm_cam.ini' \
    --debug \
    "$@"
exit
