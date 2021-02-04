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

google_xmanager launch \
    'experimental/users/xiuming/sim/sim/trainvali_xm.py' \
    -- \
    --xm_resource_pool='peace' \
    --xm_resource_alloc='group:peace/gcam' \
    --noxm_monitor_on_launch \
    "$@"
exit

google_xmanager launch \
    'experimental/users/xiuming/sim/sim/trainvali_xm.py' \
    -- \
    --xm_skip_launch_confirmation \
    --xm_resource_pool='perception' \
    --xm_resource_alloc='group:perception/gcam' \
    --noxm_monitor_on_launch \
    "$@"
exit
