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

# pylint: invalid-unary-operand-type

from os.path import basename
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil
from nerfactor.datasets.base import Dataset as BaseDataset


logger = logutil.Logger(loggee="datasets/brdf_merl")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False):
        # Get all BRDF names
        self.root = config.get('DEFAULT', 'data_root')
        train_paths = xm.os.sortglob(self.root, f'{mode}*.npz')
        self.brdf_names = [
            basename(x)[len('train_'):-len('.npz')] for x in train_paths]
        # Parent init.
        super().__init__(config, mode, debug=debug)
        # Cache the test data since it's shared by all BRDF identities;
        # this would save us from loading it over and over again
        if mode == 'test':
            paths = xm.os.sortglob(self.root, f'{mode}*.npz')
            assert len(paths) == 1, (
                "There should be a single set of test coordinates, shared by "
                "all identities")
            test_data = ioutil.load_np(paths[0])
        else:
            test_data = None
        self.test_data = test_data

    def _get_batch_size(self):
        bs = self.config.getint('DEFAULT', 'n_rays_per_step')
        return bs

    # pylint: disable=arguments-differ
    def _glob(self, seed=0, n_iden=20, n_between=11):
        if self.mode == 'test':
            # In testing, there is a single file of Rusink., and we need to
            # fake some paths, which are in fact material names
            paths = []
            # First 100: novel Rusink., but seen identities
            for id_ in self.brdf_names:
                paths.append(id_)
            # Next: novel Rusink. and interpolated identities
            np.random.seed(seed) # fix random seed
            mats = np.random.choice(self.brdf_names, n_iden, replace=False)
            i = 0
            for mat_i in range(n_iden - 1):
                mat1, mat2 = mats[mat_i], mats[mat_i + 1]
                for a in np.linspace(1, 0, n_between, endpoint=True):
                    id_ = '{i:06d}_{a:f}_{m1}_{b:f}_{m2}'.format(
                        i=i, a=a, m1=mat1, b=1 - a, m2=mat2)
                    paths.append(id_)
                    i += 1
            np.random.seed() # restore random seed
        else:
            paths = xm.os.sortglob(
                self.root, '{mode}*.npz'.format(mode=self.mode))
        logger.info("Number of '%s' identities: %d", self.mode, len(paths))
        return paths

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, i, envmap_h, ims, spp, rusink, refl = tf.py_function(
            self._load_data, [path], (
                tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32,
                tf.float32))
        return id_, i, envmap_h, ims, spp, rusink, refl

    def _load_data(self, path): # pylint: disable=arguments-differ
        if self.mode == 'test':
            id_ = path
            data = self.test_data
        else:
            path = tutil.eager_tensor_to_str(path)
            data = ioutil.load_np(path)
        envmap_h = data['envmap_h'][()]
        ims = data['ims'][()]
        spp = data['spp'][()]
        rusink = data['rusink']
        if self.mode == 'test':
            i = self.brdf_names.index(id_) if id_ in self.brdf_names else -1
            refl = np.zeros(
                (rusink.shape[0], 1), dtype=rusink.dtype) # placeholder
        else:
            id_ = data['name'][()]
            i = data['i'][()]
            refl = data['refl']
        return id_, i, envmap_h, ims, spp, rusink, refl

    # pylint: disable=arguments-differ
    def _process_example_postcache(
            self, id_, i, envmap_h, ims, spp, rusink, refl):
        """Samples BRDF table entries.
        """
        rusink, refl = self._sample_entries((rusink, refl))
        # NOTE: some memory waste below to make distributed strategy happy
        n = tf.shape(rusink)[0]
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (n,))
        i = tf.tile(tf.expand_dims(i, axis=0), (n,))
        envmap_h = tf.tile(tf.expand_dims(envmap_h, axis=0), (n,))
        ims = tf.tile(tf.expand_dims(ims, axis=0), (n,))
        spp = tf.tile(tf.expand_dims(spp, axis=0), (n,))
        return id_, i, envmap_h, ims, spp, rusink, refl

    def _sample_entries(self, tbls):
        # Assert same number of rows
        n_rows = None
        for tbl in tbls:
            if n_rows is None:
                n_rows = tf.shape(tbl)[0]
            else:
                tf.debugging.assert_equal(
                    tf.shape(tbl)[0], n_rows, message=(
                        "Tables must have the same number of rows to sample "
                        "from"))
        # Shortcircuit if need all
        if self.mode in ('vali', 'test'):
            return tbls
        # Training: sample entries
        # Use tf.random instead of np.random here so that the randomness is
        # correct even if we compile this to static graph using tf.function
        select_ind = tf.random.uniform(
            (self.bs,), minval=0, maxval=tf.shape(tbls[0])[0], dtype=tf.int32)
        select_ind = select_ind[:, None]
        ret = []
        for tbl in tbls:
            tbl_rows = tf.gather_nd(tbl, select_ind)
            ret.append(tbl_rows)
        return ret
