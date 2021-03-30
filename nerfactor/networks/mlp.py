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
from .seq import Network as BaseNetwork


logger = logutil.Logger(loggee="networks/mlp")


class Network(BaseNetwork):
    def __init__(self, widths, act=None, skip_at=None):
        super().__init__()
        depth = len(widths)
        if act is None:
            act = [None] * depth
        assert len(act) == depth, \
            "If not `None`, `act` must have the save length as `widths`"
        for w, a in zip(widths, act):
            if isinstance(a, str):
                a = tf.keras.layers.Activation(a)
            layer = tf.keras.layers.Dense(w, activation=a)
            self.layers.append(layer)
        self.skip_at = skip_at

    def __call__(self, x):
        # Shortcircuit if simply sequential
        if self.skip_at is None:
            return super().__call__(x)
        # Need to concatenate input at some levels
        x_ = x + 0 # make a copy
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if i in self.skip_at:
                y = tf.concat((y, x), -1)
            x_ = y
        return y
