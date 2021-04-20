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
    echo "Usage: $0 indir ims outdir[ ...]"
    exit 1
fi
indir="$1"
ims="$2"
outdir="$3"
shift # shift the remaining arguments
shift
shift

PYTHONPATH="$REPO_DIR" \
    python "$REPO_DIR"/data_gen/merl/make_dataset.py \
    --indir="$indir" \
    --ims="$ims" \
    --outdir="$outdir" \
    "$@"
