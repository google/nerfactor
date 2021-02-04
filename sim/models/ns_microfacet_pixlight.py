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

# pylint: disable=relative-beyond-top-level,arguments-differ

import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .ns_microfacet import Model as BaseModel
from ..util import logging as logutil


logger = logutil.Logger(loggee="models/ns_microfacet_pixlight")


class Model(BaseModel):
    @property
    def light(self):
        if self._light is None: # initialize just once
            maxv = self.config.getfloat('DEFAULT', 'light_init_max')
            light = tf.random.uniform(
                self.light_res + (3,), minval=0., maxval=maxv)
            self._light = tf.Variable(light, trainable=True)
        # No negative light
        return tf.clip_by_value(self._light, 0., np.inf) # 3D

    def compute_loss(self, pred, gt, **kwargs):
        """Additional priors on light probes.
        """
        light_tv_weight = self.config.getfloat('DEFAULT', 'light_tv_weight')
        light_achro_weight = self.config.getfloat(
            'DEFAULT', 'light_achro_weight')
        loss = 0
        mode = kwargs['mode'] # don't pop, since it's still needed
        if mode == 'train':
            light = self.light
            # Spatial TV penalty
            if light_tv_weight > 0:
                dx = light - tf.roll(light, 1, 1)
                dy = light - tf.roll(light, 1, 0)
                tv = tf.reduce_sum(dx ** 2 + dy ** 2)
                loss += light_tv_weight * tv
            # Across-channel TV penalty
            if light_achro_weight > 0:
                dc = light - tf.roll(light, 1, 2)
                tv = tf.reduce_sum(dc ** 2)
                loss += light_achro_weight * tv
        # Other losses
        loss += super().compute_loss(pred, gt, **kwargs)
        return loss
