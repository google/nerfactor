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
from .base import Network as BaseNetwork


logger = logutil.Logger(loggee="networks/seq")


class Network(BaseNetwork):
    """Assuming simple sequential flow.
    """
    def build(self, input_shape):
        seq = tf.keras.Sequential(self.layers)
        seq.build(input_shape)
        for layer in self.layers:
            assert layer.built, "Some layers not built"

    def __call__(self, tensor):
        x = tensor
        for layer in self.layers:
            y = layer(x)
            x = y
        return y
