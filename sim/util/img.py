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

# pylint: disable=relative-beyond-top-level,invalid-unary-operand-type

import numpy as np
from PIL import Image

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from . import logging as logutil


logger = logutil.Logger(loggee="util/img")


class GaussianBlur():
    """Mainly to support Gaussian blurring before computing L1, a feature that
    might alleviate the color shift problem when L1 is used together with a
    perceptual loss.

    Intuition: "Consider the limiting case: if you blurred all the way (take
    global average of the image) and set the weight on the blurred-L1 to
    infinity, it would match the mean RGB value of the image."
    """
    def __init__(self, sigma, kernel_size=None, n_ch=3):
        if kernel_size is None:
            kernel_size = 6 * sigma # +/- 3 sigmas covers 99.7%
        x = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(x, x)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[:, :, tf.newaxis], (1, 1, n_ch)) # KxKxC
        kernel = kernel[:, :, :, tf.newaxis] # KxKxCx1
        self.kernel = kernel

    def filter(self, x):
        """Input shape should be NxHxWxC.
        """
        y = tf.nn.depthwise_conv2d(
            x, self.kernel, (1, 1, 1, 1), 'SAME', data_format='NHWC')
        return y


def _clip_0to1_warn(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, tf.Tensor):
        if tf.reduce_min(tensor_0to1) < 0 or tf.reduce_max(tensor_0to1) > 1:
            logger.debug(msg)
            tensor_0to1 = tf.clip_by_value(
                tensor_0to1, clip_value_min=0, clip_value_max=1)
    else:
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            logger.debug(msg)
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    return tensor_0to1


def alpha_blend(tensor1, alpha, tensor2=None):
    """Alpha-blend two tensors. If the second tensor is `None`, the first
    tensor will be blended with a all-zero tensor, equivalent to masking if
    alpha is binary.

    Alpha should range from 0 to 1.
    """
    if isinstance(tensor1, tf.Tensor):
        mod = tf
    else:
        mod = np

    if tensor2 is None:
        tensor2 = mod.zeros_like(tensor1)

    if len(mod.shape(tensor1)) == 3 and len(mod.shape(alpha)) == 2:
        alpha = mod.reshape(alpha, tuple(mod.shape(alpha)) + (1,))
        alpha = mod.tile(alpha, (1, 1, mod.shape(tensor1)[2]))

    return mod.multiply(tensor1, alpha) + mod.multiply(tensor2, 1. - alpha)


def resize(img, new_h=None, new_w=None):
    """Uses TensorFlow's bilinear, antialiasing resizing, accepting an
    HxWxC tensor or array.

    Compatible with graph execution.
    """
    if isinstance(img, tf.Tensor):
        input_is_tensor = True
        tensor = img
    else:
        input_is_tensor = False
        tensor = tf.convert_to_tensor(img)

    # Original size
    hw = tf.shape(tensor)
    h, w = hw[0], hw[1] # both 0D tensors

    # Figure out new size
    if new_h is not None and new_w is not None:
        if not tf.equal(tf.cast(h / w * new_w, h.dtype), new_h):
            logger.warn((
                "Aspect ratio changed in resizing: original size is %s; "
                "new size is %s"), (h, w), (new_h, new_w))
    elif new_h is None and new_w is not None:
        new_h = tf.cast(h / w * new_w, h.dtype)
    elif new_h is not None and new_w is None:
        new_w = tf.cast(w / h * new_h, h.dtype)
    else:
        raise ValueError("At least one of new height or width must be given")
    new_shape = (new_h, new_w)

    resized = tf.image.resize(
        tensor, new_shape, method='bilinear', antialias=True)

    if input_is_tensor:
        return resized

    # Restore the original data type if input is not a tensor
    orig_dtype = img.dtype if isinstance(img, np.ndarray) else img.numpy().dtype
    return resized.numpy().astype(orig_dtype)


def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, tf.Tensor):
        pow_func = tf.math.pow
        where_func = tf.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = _clip_0to1_warn(tensor_0to1)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def to_uint(tensor_0to1, target_type='uint8'):
    """Converts a float tensor/array with values in [0, 1] to an unsigned
    integer type for visualization.
    """
    if isinstance(tensor_0to1, tf.Tensor):
        target_type = tf.as_dtype(target_type)
        tensor_0to1 = _clip_0to1_warn(tensor_0to1)
        tensor_uint = tf.cast(tensor_0to1 * target_type.max, target_type)
    else:
        tensor_0to1 = _clip_0to1_warn(tensor_0to1)
        tensor_uint = (np.iinfo(target_type).max * tensor_0to1).astype(
            target_type)
    return tensor_uint


# NOTE: untested
def rot90(img, counterclockwise=False):
    if isinstance(img, np.ndarray):
        from_to = (0, 1) if counterclockwise else (1, 0)
        img_ = np.rot90(img, axes=from_to)
    elif isinstance(img, tf.Tensor):
        k = 1 if counterclockwise else 3
        img_ = tf.image.rot90(img, k=k)
    else:
        raise TypeError(img)
    return img_


def set_left_top_corner(tensor, val):
    """Thanks to "eager tensor doesn't support assignment."
    """
    mask = np.ones(tensor.shape)
    mask[:, 0, 0, :] = val
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    return tf.multiply(mask, tensor)


def hconcat(img_list, out_w=None):
    total = []
    for img in img_list:
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        if total:
            prev = total[-1]
            img = resize(img, new_h=prev.shape[0])
        total.append(img)
    total = np.hstack(total)
    if out_w is not None:
        total = resize(total, new_w=out_w)
    return total


def vconcat(img_list, out_h=None):
    total = []
    for img in img_list:
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        if total:
            prev = total[-1]
            img = resize(img, new_w=prev.shape[1])
        total.append(img)
    total = np.vstack(total)
    if out_h is not None:
        total = resize(total, new_h=out_h)
    return total


def frame_image(img, rgb=None, width=4):
    if rgb is None:
        rgb = (0, 0, 1)
    rgb = np.array(rgb, dtype=img.dtype) * np.iinfo(img.dtype).max

    img[:width, :, :] = rgb
    img[-width:, :, :] = rgb
    img[:, :width, :] = rgb
    img[:, -width:, :] = rgb


def embed_into(inset, img, inset_scale=0.2):
    inset_h = int(inset_scale * img.shape[0])
    inset_w = int(inset_h / inset.size[1] * inset.size[0])
    inset = inset.resize((inset_w, inset_h))
    bg = Image.fromarray(img)
    bg.paste(inset, (bg.size[0] - inset.size[0], 0), inset) # inset's A
    # channel will be used as mask
    composite = np.array(bg)
    return composite
