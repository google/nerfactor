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

from . import logging as logutil


logger = logutil.Logger(loggee="util/tensor")


def shape_as_list(x):
    return x.get_shape().as_list()


def make_nhwc(batch, c=3):
    """Makes a NxHxW(x1) tensor NxHxWxC, written in graph mode.
    """
    # Assert 3D or 4D
    n_dims = tf.rank(batch)
    assert_op = tf.debugging.Assert(
        tf.logical_or(tf.equal(n_dims, 3), tf.equal(n_dims, 4)), [n_dims])

    # If necessary, 3D to 4D
    with tf.control_dependencies([assert_op]):
        batch = tf.cond(
            tf.equal(n_dims, 4),
            true_fn=lambda: batch,
            false_fn=lambda: tf.expand_dims(batch, -1))

    # Repeat the last channel Cx, after asserting #channels is 1
    shape = tf.shape(batch)
    assert_op = tf.debugging.Assert(tf.equal(shape[3], 1), [shape])
    with tf.control_dependencies([assert_op]):
        return tf.tile(batch, (1, 1, 1, c))


def eager_tensor_to_str(x):
    if isinstance(x, str): # so that the operation is idempotent
        return x
    return x.numpy().decode()


def one_hot_img(h, w, c, i, j):
    """Makes a float32 HxWxC tensor with 1s at (i, j, *) and 0s everywhere else.
    """
    ind = [(i, j, x) for x in range(c)]
    ind = tf.convert_to_tensor(ind)
    updates = tf.ones((c,), dtype=tf.float32)
    one_hot = tf.scatter_nd(ind, updates, (h, w, c))
    return one_hot
