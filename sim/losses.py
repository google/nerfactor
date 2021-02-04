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

# pylint: disable=relative-beyond-top-level

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.third_party.google_research.google_research.robust_loss \
    import adaptive

from .util import img as imgutil


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


class Barron():
    def __init__(self, imw, imh):
        alpha = 1 # fix to Charbonnier loss for now, as that usually works
        # well, and is similar to L1
        scale = 0.01 # because pixels are in [0, 1]
        wavelet_scale_base = 1 # this hyperparameter can have a huge effect
        # in how low-frequency errors are weighted against high-frequency
        # errors. Try setting this to 0.5 and 2 as well, and see what works
        # the best
        self.func = adaptive.AdaptiveImageLossFunction(
            (imh, imw, 3), tf.float32,
            color_space='YUV', representation='CDF9/7',
            summarize_loss=False,
            wavelet_num_levels=5, wavelet_scale_base=wavelet_scale_base,
            alpha_lo=alpha, alpha_hi=alpha,
            scale_lo=scale, scale_init=scale)

    def __call__(self, gt, pred, weights=None):
        if weights is not None:
            gt = imgutil.alpha_blend(gt, weights)
            pred = imgutil.alpha_blend(pred, weights)
        loss = self.func(gt - pred) # NxHxWxC
        loss = tf.reduce_mean(loss) # scalar
        return loss


class LPIPS():
    weight_f = (
        '/cns/ok-d/home/gcam-eng/gcam/interns/xiuming/relight/data/lpips'
        '/net-lin_alex_v0.1.pb')

    def __init__(self, per_ch=False):
        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.compat.v1.import_graph_def(graph_def, name="")
            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))
        if not hasattr(self, 'func'):
            # Pack LPIPS network into a tf function
            graph_def = tf.compat.v1.GraphDef()
            with gfile.Open(self.weight_f, 'rb') as h:
                graph_def.ParseFromString(h.read())
            self.func = tf.function(wrap_frozen_graph(
                graph_def, inputs=['0:0', '1:0'], outputs='Reshape_10:0'))
        self.per_ch = per_ch

    def __call__(self, gt, pred, weights=None):
        """Inputs should be in [0, 1] and have shape NxHxWxC.
        """
        assert gt.shape[3] == 3 and pred.shape[3] == 3, \
            "Both ground truth and prediction must be of shape `(N, H, W, 3)`"
        if weights is not None:
            gt = imgutil.alpha_blend(gt, weights)
            pred = imgutil.alpha_blend(pred, weights)
        # [0, 1] to [-1, 1]
        gt = gt * 2 - 1
        pred = pred * 2 - 1
        # NxHxWxC to NxCxHxW
        pred = tf.transpose(pred, [0, 3, 1, 2])
        gt = tf.transpose(gt, [0, 3, 1, 2])
        if self.per_ch:
            loss = tf.zeros((pred.shape[0], 1, 1, 1))
            for i in range(3):
                pred_ch = tf.tile(pred[:, i:(i + 1), :, :], (1, 3, 1, 1))
                gt_ch = tf.tile(gt[:, i:(i + 1), :, :], (1, 3, 1, 1))
                loss += self.func(pred_ch, gt_ch) / 3 # Nx1x1x1
        else:
            loss = self.func(pred, gt) # Nx1x1x1
        loss = tf.reduce_mean(loss)
        return loss
