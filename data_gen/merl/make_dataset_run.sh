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

if [ $# -lt 3 ]; then
    echo "Usage: $0 merl_dir ims out_dir[ ...]"
    exit 1
fi
merl_dir="$1"
ims="$2"
out_dir="$3"
shift # shift the remaining arguments
shift
shift

PYTHONPATH="$REPO_DIR" \
    python make_dataset.py \
    --merl_dir="$merl_dir" \
    --envmap_h='16' \
    --ims="$ims" \
    --spp='1' \
    --out_dir="$out_dir" \
    "$@"
