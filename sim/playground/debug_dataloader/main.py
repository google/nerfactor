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

from os.path import join, basename, dirname
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from absl import app, flags

from google3.pyglib import gfile
from google3.experimental.users.xiuming.sim.sim import datasets, models
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil, math as mathutil, geom as geomutil
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string('config_ini', '', "path to configuration file")
FLAGS = flags.FLAGS

logger = logutil.Logger()


def main(_):
    mode = 'test'

    config = ioutil.read_config(FLAGS.config_ini)

    # Make dataset
    dataset_name = config.get('DEFAULT', 'dataset')
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, mode)

    ret = dataset._load_data(dataset.files[0])

    args = []
    for x in ret:
        args.append(tf.convert_to_tensor(x))
    ret = dataset._sample_rays(*args[1:])
    # ret = dataset._sample_entries(args[-2:])

    # Run inference on a single batch
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)
    for batch in datapipe:
        from IPython import embed; embed()


if __name__ == '__main__':
    app.run(main)
