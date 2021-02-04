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
tf.compat.v1.enable_eager_execution()

from ..util import logging as logutil


logger = logutil.Logger(loggee="datasets/base")


class Dataset():
    def __init__(
            self, config, mode, debug=False, shuffle_buffer_size=64,
            prefetch_buffer_size=None, n_map_parallel_calls=None):
        assert mode in ('train', 'vali', 'test'), (
            "Accepted dataset modes: 'train', 'vali', 'test', but input is %s"
        ) % mode
        self.config = config
        self.mode = mode
        self.debug = debug
        if debug:
            logger.warn("Dataset in debug mode; behavior may be different")
        self.shuffle_buffer_size = shuffle_buffer_size
        if prefetch_buffer_size is None:
            prefetch_buffer_size = tf.data.experimental.AUTOTUNE
        self.prefetch_buffer_size = prefetch_buffer_size
        if n_map_parallel_calls is None:
            n_map_parallel_calls = tf.data.experimental.AUTOTUNE
        self.n_map_parallel_calls = n_map_parallel_calls
        self.files = self._glob()
        assert self.files, "No files to process into a dataset"
        self.bs = self._get_batch_size()

    def _glob(self):
        """Globs the source data files (like paths to images), each of which
        will be processed by the processing functions below.

        Returns:
            list(str): List of paths to the data files.
        """
        raise NotImplementedError

    def _get_batch_size(self):
        """Useful for NeRF-like models, where the effective batch size may not
        be just number of images, and for models where different modes have
        different batch sizes.

        Returns:
            int: Batch size.
        """
        if 'bs' not in self.config['DEFAULT'].keys():
            raise ValueError((
                "Specify batch size either as 'bs' in the configuration file, "
                "or override this function to generate a value another way"))
        return self.config.getint('DEFAULT', 'bs')

    def _process_example_precache(self, path):
        """Output of this function will be cached.
        """
        raise NotImplementedError

    # pylint: disable=no-self-use
    def _process_example_postcache(self, *args):
        """Move whatever you don't want cached into this function, such as
        processing that involves randomness.

        If you don't override this, this will be a no-op.
        """
        return args

    def build_pipeline(
            self, filter_predicate=None, seed=None, no_batch=False,
            no_shuffle=False):
        cache = self.config.getboolean('DEFAULT', 'cache')
        is_train = self.mode == 'train'
        # Make dataset from files
        files = sorted(self.files)
        dataset = tf.data.Dataset.from_tensor_slices(files)
        # Optional filtering
        if filter_predicate is not None:
            dataset = dataset.filter(filter_predicate)
        # Parallelize processing
        dataset = dataset.map(
            self._process_example_precache,
            num_parallel_calls=self.n_map_parallel_calls)
        if cache:
            dataset = dataset.cache()
        # Useful if part of your processing involves randomness
        dataset = dataset.map(
            self._process_example_postcache,
            num_parallel_calls=self.n_map_parallel_calls)
        # Shuffle
        if is_train and (not no_shuffle):
            dataset = dataset.shuffle(self.shuffle_buffer_size, seed=seed)
        # Batching
        if not no_batch:
            # In case you want to make batches yourself
            dataset = dataset.batch(batch_size=self.bs)
        # Prefetching
        datapipe = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        return datapipe
