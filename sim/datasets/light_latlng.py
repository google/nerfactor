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

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .brdf_merl import Dataset as BaseDataset
from ..util import logging as logutil, io as ioutil, tensor as tutil


logger = logutil.Logger(loggee="datasets/light_latlng")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False):
        super().__init__(config, mode, debug=debug)
        self.light_names = self.brdf_names
        delattr(self, 'brdf_names')

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, i, envmap_h, latlng, radi = tf.py_function(
            self._load_data, [path], (
                tf.string, tf.int32, tf.int32, tf.float32, tf.float32))
        return id_, i, envmap_h, latlng, radi

    def _load_data(self, path, eps=1e-16): # pylint: disable=arguments-differ
        if self.mode == 'test':
            id_ = path
            data = self.test_data
        else:
            path = tutil.eager_tensor_to_str(path)
            data = ioutil.load_np(path)
        envmap_h = data['envmap_h'][()]
        latlng = data['latlng']
        if self.mode == 'test':
            i = self.light_names.index(id_) if id_ in self.light_names else -1
            radi = np.zeros(
                (latlng.shape[0], 1), dtype=latlng.dtype) # placeholder
        else:
            id_ = data['name'][()]
            i = data['i'][()]
            radi = data['radi']
        # Prevent exactly 0 radiance, which is problematic for log
        radi = np.clip(radi, eps, np.inf)
        return id_, i, envmap_h, latlng, radi

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, i, envmap_h, latlng, radi):
        """Samples light table entries.
        """
        latlng, radi = self._sample_entries((latlng, radi))
        # NOTE: some memory waste below to make distributed strategy happy
        n = tf.shape(latlng)[0]
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (n,))
        i = tf.tile(tf.expand_dims(i, axis=0), (n,))
        envmap_h = tf.tile(tf.expand_dims(envmap_h, axis=0), (n,))
        return id_, i, envmap_h, latlng, radi
