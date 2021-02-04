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


printf "~~~~~~~~~~~~~~~~\n"
printf "Run, on this laptop,\n"
printf "\tssh -N -f -L 8888:127.0.0.1:8888 xiuming@xiuming.com\n"
printf "for port forwarding\n"
printf "~~~~~~~~~~~~~~~~\n"

blaze run -c opt \
    'experimental/users/xiuming/sim/sim/playground/normals_from_volume' \
    -- \
    "$@"
