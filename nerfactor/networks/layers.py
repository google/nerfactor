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
from tensorflow import keras

from util import logging as logutil, geom as geomutil, math as mathutil


logger = logutil.Logger(loggee="networks/layers")


class LatentCode(keras.layers.Layer):
    """Latent code to be optimized, as in Generative Latent Optimization.
    """
    def __init__(self, n_iden, dim, mean=0., std=1., normalize=False):
        super(LatentCode, self).__init__()
        init = tf.random_normal_initializer(mean=mean, stddev=std)
        z = init(shape=(n_iden, dim), dtype='float32')
        self._z = tf.Variable(initial_value=z, trainable=True)
        self.normalize = normalize

    @property
    def z(self):
        """The exposed interface for retrieving the current latent codes.
        """
        if self.normalize:
            return mathutil.safe_l2_normalize(self._z, axis=1)
        return self._z

    @z.setter
    def z(self, value):
        self._z = tf.Variable(initial_value=value, trainable=True)

    def call(self, ind):
        """When you need only some slices of z.
        """
        ind = tf.convert_to_tensor(ind)
        # 0D to 1D
        if len(tf.shape(ind)) == 0: # pylint: disable=len-as-condition
            ind = tf.reshape(ind, (1,))
        # Take values by index
        z = tf.gather_nd(self.z, ind[:, None])
        return z

    def interp(self, w1, i1, w2, i2):
        z1, z2 = self(i1), self(i2)
        if self.normalize:
            # Latent codes on unit sphere -- slerp
            assert w1 + w2 == 1., \
                "When latent codes are normalized, use weights that sum to 1"
            z = geomutil.slerp(z1, z2, w2)
        else:
            # Not normalized, can just linearly interpolate
            z = w1 * z1 + w2 * z2
        return z


def conv(kernel_size, n_ch_out, stride=1):
    return tf.keras.layers.Conv2D(
        n_ch_out,
        kernel_size,
        strides=stride,
        padding='same')


def deconv(kernel_size, n_ch_out, stride=1):
    return tf.keras.layers.Conv2DTranspose(
        n_ch_out,
        kernel_size,
        strides=stride,
        padding='same')


def upconv(n_ch_out):
    """2x upsampling a feature map.
    """
    layers = [
        tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'),
        tf.keras.layers.Conv2D(n_ch_out, 2, padding='same')]
    return tf.keras.Sequential(layers)


def norm(type_):
    if type_ == 'batch':
        norm_layer = tf.keras.layers.BatchNormalization(
            momentum=0.99, epsilon=0.001)
    elif type_ == 'layer':
        norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=0.001, center=True, scale=True)
    elif type_ == 'instance':
        norm_layer = instancenorm()
    elif type_ == 'pixel':
        norm_layer = pixelnorm()
    elif type_ is None or type_ == 'none':
        norm_layer = iden()
    else:
        raise NotImplementedError(type_)
    return norm_layer


def act(type_):
    if type_ == 'relu':
        act_layer = tf.keras.layers.ReLU(negative_slope=0)
    elif type_ == 'leakyrelu':
        act_layer = tf.keras.layers.LeakyReLU(alpha=0.3)
    elif type_ == 'elu':
        act_layer = tf.keras.layers.ELU(alpha=1.0)
    else:
        raise NotImplementedError(type_)
    return act_layer


def pool(type_):
    kwargs = {
        'pool_size': 2,
        'strides': 2,
        'padding': 'same'}
    if type_ == 'max':
        pool_layer = tf.keras.layers.MaxPooling2D(**kwargs)
    elif type_ == 'avg':
        pool_layer = tf.keras.layers.AveragePooling2D(**kwargs)
    elif type_ is None or type_ == 'none':
        pool_layer = iden()
    else:
        raise NotImplementedError(type_)
    return pool_layer


def instancenorm():
    return tf.keras.layers.Lambda(
        lambda x: tf.contrib.layers.instance_norm(
            x, center=True, scale=True, epsilon=1e-06))


def pixelnorm():
    def _pixelnorm(images, epsilon=1.0e-8):
        """Pixel normalization.

        For each pixel a[i,j,k] of image in HWC format, normalize its
        value to b[i,j,k] = a[i,j,k] / SQRT(SUM_k(a[i,j,k]^2) / C + eps).

        Args:
            images: A 4D `Tensor` of NHWC format.
            epsilon: A small positive number to avoid division by zero.

        Returns:
            A 4D `Tensor` with pixel-wise normalized channels.
        """
        return images * tf.rsqrt(
            tf.reduce_mean(
                tf.square(images), axis=3, keepdims=True
            ) + epsilon)
    return tf.keras.layers.Lambda(_pixelnorm)


def iden():
    return tf.keras.layers.Lambda(lambda x: x)
