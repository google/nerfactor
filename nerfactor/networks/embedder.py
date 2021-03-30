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

from util import logging as logutil


logger = logutil.Logger(loggee="networks/embedder")


class Embedder:
    def __init__(
            self, incl_input=True, in_dims=3, log2_max_freq=3, n_freqs=4,
            log_sampling=True, periodic_func=None):
        if periodic_func is None:
            periodic_func = [tf.math.sin, tf.math.cos]
        embed_func = []
        out_dims = 0
        if incl_input:
            embed_func.append(lambda x: x)
            out_dims += in_dims
        if log_sampling:
            freq_bands = 2. ** tf.linspace(0., log2_max_freq, n_freqs)
        else:
            freq_bands = tf.linspace(2. ** 0., 2. ** log2_max_freq, n_freqs)
        for freq in freq_bands:
            for p_f in periodic_func:
                embed_func.append(
                    lambda x, p_f=p_f, freq=freq: p_f(x * freq))
                out_dims += in_dims
        self.out_dims = out_dims
        self.embed_func = embed_func

    def __call__(self, x):
        return tf.concat([f(x) for f in self.embed_func], -1)
