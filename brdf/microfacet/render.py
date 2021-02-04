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

from os.path import join
import numpy as np
from absl import app, flags

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.microfacet.microfacet \
    import Microfacet
from google3.experimental.users.xiuming.sim.brdf.renderer import SphereRenderer


flags.DEFINE_float('default_rough', 0.3, "")
flags.DEFINE_boolean('lambert_only', False, "")
flags.DEFINE_string('envmap_path', '', "")
flags.DEFINE_integer('envmap_h', 16, "")
flags.DEFINE_float('envmap_inten', 1., "")
flags.DEFINE_integer('ims', 256, "")
flags.DEFINE_integer('spp', 1, "")
flags.DEFINE_string('out_dir', '', "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    brdf = Microfacet(
        default_rough=FLAGS.default_rough, lambert_only=FLAGS.lambert_only)

    renderer = SphereRenderer(
        FLAGS.envmap_path, FLAGS.out_dir, envmap_inten=FLAGS.envmap_inten,
        envmap_h=FLAGS.envmap_h, ims=FLAGS.ims, spp=FLAGS.spp,
        debug=FLAGS.debug)

    # Light directions
    pts2l = renderer.gen_light_dir(local=False)
    pts2l = np.reshape(pts2l, (-1, pts2l.shape[2], 3))

    # Viewing directions
    pts2c = renderer.gen_view_dir(local=False)
    pts2c = np.reshape(pts2c, (-1, 3))

    # Normals
    normal = np.reshape(renderer.normal, (-1, 3))

    # Query BRDF
    pts2l = tf.convert_to_tensor(pts2l, dtype=tf.float32)
    pts2c = tf.convert_to_tensor(pts2c, dtype=tf.float32)
    normal = tf.convert_to_tensor(normal, dtype=tf.float32)
    brdf_val = brdf(pts2l, pts2c, normal)
    brdf_val = tf.reshape(brdf_val, renderer.lcontrib.shape)

    # Render
    render = renderer.render(brdf_val)
    out_path = join(FLAGS.out_dir, 'render.png')
    xm.io.img.write_arr(render, out_path, clip=True)


if __name__ == '__main__':
    app.run(main)
