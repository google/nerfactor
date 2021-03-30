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


def log10(x):
    num = tf.math.log(x)
    denom = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / denom


@tf.custom_gradient
def safe_atan2(x, y, eps=1e-6):
    """Numerically stable version to safeguard against (0, 0) input, which
    causes the backward of tf.atan2 to go NaN.
    """
    z = tf.atan2(x, y)

    def grad(dz):
        denom = x ** 2 + y ** 2
        denom += eps
        dzdx = y / denom
        dzdy = -x / denom
        return dz * dzdx, dz * dzdy

    return z, grad


@tf.custom_gradient
def safe_acos(x, eps=1e-6):
    """Numerically stable version to safeguard against +/-1 input, which
    causes the backward of tf.acos to go inf.

    Simply clipping the input at +/-1-/+eps gives 0 gradient at the clipping
    values, but analytically, the gradients there should be large.
    """
    x_clip = tf.clip_by_value(x, -1., 1.)
    y = tf.acos(x_clip)

    def grad(dy):
        in_sqrt = 1. - x_clip ** 2
        in_sqrt += eps
        denom = tf.sqrt(in_sqrt)
        denom += eps
        dydx = -1. / denom
        return dy * dydx

    return y, grad


def safe_l2_normalize(x, axis=None, eps=1e-6):
    return tf.linalg.l2_normalize(x, axis=axis, epsilon=eps)


def safe_cumprod(x, eps=1e-6):
    return tf.math.cumprod(x + eps, axis=-1, exclusive=True)


def inv_transform_sample(val, weights, n_samples, det=False, eps=1e-5):
    denom = tf.reduce_sum(weights, -1, keepdims=True)
    denom += eps
    pdf = weights / denom
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat((tf.zeros_like(cdf[:, :1]), cdf), -1)

    if det:
        u = tf.linspace(0., 1., n_samples)
        u = tf.broadcast_to(u, cdf.shape[:-1] + (n_samples,))
    else:
        u = tf.random.uniform(cdf.shape[:-1] + (n_samples,))

    ind = tf.searchsorted(cdf, u, side='right') # (n_rays, n_samples)
    below = tf.maximum(0, ind - 1)
    above = tf.minimum(ind, cdf.shape[-1] - 1)
    ind_g = tf.stack((below, above), -1) # (n_rays, n_samples, 2)
    cdf_g = tf.gather(cdf, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    val_g = tf.gather(val, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0] # (n_rays, n_samples)
    denom = tf.where(denom < eps, tf.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom
    samples = val_g[:, :, 0] + t * (val_g[:, :, 1] - val_g[:, :, 0])
    return samples # (n_rays, n_samples)
