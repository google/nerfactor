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
from tqdm import tqdm
from absl import app, flags

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.microfacet.microfacet \
    import Microfacet


flags.DEFINE_string('out_dir', '', "")
flags.DEFINE_boolean('skip_npy', False, "")
FLAGS = flags.FLAGS


def main(_):
    n_rough = 10
    n_lat = 10
    lat_i = 7

    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    xyz = np.array((0, 0, 0))
    normal = np.array((0, 0, 1))

    # Viewing directions
    r = 1 * np.ones((n_lat, 1))
    lng = 0 * np.ones((n_lat, 1))
    lat = np.linspace(0, np.pi, n_lat, endpoint=False).reshape((-1, 1))
    rlatlng = np.hstack((r, lat, lng))
    cam_locs = xm.geometry.sph.sph2cart(rlatlng)
    pts2c = cam_locs - xyz
    pts2c = xm.linalg.normalize(pts2c, axis=1)
    if not FLAGS.skip_npy:
        with gfile.Open(join(FLAGS.out_dir, 'pts2c.npy'), 'wb') as h:
            np.save(h, pts2c)
    pts2c = tf.convert_to_tensor(pts2c, dtype=tf.float32)

    # Light directions
    pts2l = cam_locs[lat_i, :] - xyz
    pts2l = np.tile(pts2l[None, None, :], (pts2c.shape[0], 1, 1))
    pts2l = xm.linalg.normalize(pts2l, axis=2)
    if not FLAGS.skip_npy:
        with gfile.Open(join(FLAGS.out_dir, 'pts2l.npy'), 'wb') as h:
            np.save(h, pts2l)
    pts2l = tf.convert_to_tensor(pts2l, dtype=tf.float32)

    # Normals
    normal = np.tile(normal[None, :], (pts2c.shape[0], 1))
    normal = xm.linalg.normalize(normal, axis=1)
    if not FLAGS.skip_npy:
        with gfile.Open(join(FLAGS.out_dir, 'normal.npy'), 'wb') as h:
            np.save(h, normal)
    normal = tf.convert_to_tensor(normal, dtype=tf.float32)

    xy = np.array(lat)

    rough_vals = np.linspace(0, 1, n_rough, endpoint=True)
    for rough in tqdm(rough_vals, desc="Roughness Values"):
        brdf = Microfacet(default_rough=rough)

        # Query BRDF
        brdf_val = brdf(pts2l, pts2c, normal)
        brdf_val = brdf_val.numpy()[:, :, 0] # since all channels are the same
        if not FLAGS.skip_npy:
            with gfile.Open(
                    join(FLAGS.out_dir, 'brdf_rough%.2f.npy' % rough),
                    'wb') as h:
                np.save(h, brdf_val)

        log_brdf_val = np.log10(brdf_val)
        xy = np.hstack((xy, log_brdf_val))

    labels = ['%.2f' % x for x in rough_vals]
    plot = xm.vis.plot.Plot(
        labels=labels, outpath=join(FLAGS.out_dir, 'brdf.png'),
        xlabel="Viewing Theta", ylabel="Log BRDF",
        figtitle="Normal: +Z; Light Theta: %.2f" % lat[lat_i, 0])
    plot.line(xy, width=5, marker='o', marker_size=10)


if __name__ == '__main__':
    app.run(main)
