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

# pylint: disable=invalid-unary-operand-type

import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from . import logging as logutil, img as imgutil


logger = logutil.Logger(loggee="util/light")


def vis_light(light_probe, outpath=None, h=None):
    # In case we are predicting too low of a resolution
    if h is not None:
        light_probe = imgutil.resize(light_probe, new_h=h)

    # We need NumPy array onwards
    if isinstance(light_probe, tf.Tensor):
        light_probe = light_probe.numpy()

    # Tonemap
    img = xm.img.tonemap(light_probe, method='gamma', gamma=4) # [0, 1]
    # srgb = xm.img.linear2srgb(linear)
    # srgb_uint = xm.img.denormalize_float(srgb)
    img_uint = xm.img.denormalize_float(img)

    # Optionally, write to disk
    if outpath is not None:
        xm.io.img.write_img(img_uint, outpath)

    return img_uint
