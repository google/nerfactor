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

import tensorflow as tf

from util import img as imgutil


class L1():
    def __init__(self):
        self.func = tf.keras.losses.MeanAbsoluteError(reduction='none')

    def __call__(self, gt, pred, weights=None):
        loss = self.func(gt, pred, sample_weight=weights)
        # Because of no reduction, we have a NxHxW tensor here
        loss = tf.reduce_mean(loss)
        # Averaged across pixels and samples, so a scalar
        return loss


class L2():
    def __init__(self):
        self.func = tf.keras.losses.MeanSquaredError(reduction='none')

    def __call__(self, gt, pred, keep_batch=False, weights=None):
        loss = self.func(gt, pred, sample_weight=weights)
        # Because of no reduction, we lose only the last dimension here
        if keep_batch:
            reduce_axes = tuple(range(len(loss.shape)))[1:]
            loss = tf.reduce_mean(loss, axis=reduce_axes)
            # Averaged across every dimension but batch, so a (N,) tensor
        else:
            loss = tf.reduce_mean(loss, axis=None)
            # Averaged across every dimension (including batch), so a scalar
        return loss


class UVL2():
    """UV as in YUV, not UV space.
    """
    def __init__(self):
        self.func = tf.keras.losses.MeanSquaredError(reduction='none')

    def __call__(self, gt, pred, weights=None):
        gt_clip = tf.clip_by_value(gt, 0, 1)
        pred_clip = tf.clip_by_value(pred, 0, 1)
        gt_yuv = tf.image.rgb_to_yuv(gt_clip)
        pred_yuv = tf.image.rgb_to_yuv(pred_clip)
        loss = self.func(
            gt_yuv[..., 1:], pred_yuv[..., 1:], sample_weight=weights)
        # Because of no reduction, we have a NxHxW tensor here
        loss = tf.reduce_mean(loss)
        # Averaged across pixels and samples, so a scalar
        return loss


class SSIM():
    def __init__(self, dynamic_range):
        self.func = tf.image.ssim
        self.dynamic_range = dynamic_range # i.e., max - min

    def __call__(self, gt, pred, weights=None):
        if weights is not None:
            gt = imgutil.alpha_blend(gt, weights)
            pred = imgutil.alpha_blend(pred, weights)
        sim = self.func(gt, pred, self.dynamic_range) # N-vector, [-1, 1]
        loss = (1 - sim) / 2 # N-vector, [0, 1]
        loss = tf.reduce_mean(loss) # a scalar
        return loss
