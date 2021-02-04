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


dataset="$1"

nfs_dir='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3'
local_dir='/usr/local/home/xiuming/Desktop'
cns_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021'

set -e

# Delete the stale
rm -f "$local_dir/$dataset.zip"
rm -rf "$local_dir/$dataset"
fileutil rm -R -f --gfs_user='gcam-gpu' "$cns_dir/$dataset"

# Download
wget \
    "http://vision38.csail.mit.edu$nfs_dir/$dataset.zip" \
    --directory-prefix="$local_dir/"

# Unzip
unzip \
    "$local_dir/$dataset.zip" \
    -d "$local_dir/$dataset"

# Copy
fileutil cp -colossus_parallel_copy -parallel_copy=8 -R \
    "$local_dir/$dataset" \
    "$cns_dir/"
