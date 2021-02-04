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

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.merl.merl import MERL


flags.DEFINE_integer('n_light_dir', 100 ** 2 // 2, "")
flags.DEFINE_integer('n_view_dir', 100 ** 2 // 2, "")
flags.DEFINE_string('merl_dir', '', "")
flags.DEFINE_string('out_dir', '', "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    n = np.array((0, 0, 1), dtype=float)

    # Generate viewing directions
    rlatlng = xm.geometry.sph.uniform_sample_sph(2 * FLAGS.n_view_dir)
    rlatlng = rlatlng[rlatlng[:, 1] >= 0, :] # upper hemisphere only
    xyz = xm.geometry.sph.sph2cart(rlatlng)
    vdir = xm.linalg.normalize(xyz, axis=1)
    vdir = np.reshape(vdir, (1, -1, 3)) # HxWx3

    # Figure out solid angle weights for integration
    colat = np.pi / 2 - rlatlng[:, 1]
    sin_colat = np.sin(colat)
    solid_angles = 4 * np.pi * sin_colat / np.sum(sin_colat)

    # Generate light directions
    rlatlng = xm.geometry.sph.uniform_sample_sph(2 * FLAGS.n_light_dir)
    rlatlng = rlatlng[rlatlng[:, 1] >= 0, :] # upper hemisphere only
    xyz = xm.geometry.sph.sph2cart(rlatlng)
    ldir = xm.linalg.normalize(xyz, axis=1)
    ldir = np.reshape(ldir, (1, 1, -1, 3))
    ldir_rep = np.tile(ldir, vdir.shape[:2] + (1, 1)) # HxWxLx3

    # Cosine
    cos = np.einsum('ijkl,l->ijk', ldir_rep, n) # HxWxL

    for brdf_path in tqdm(xm.os.sortglob(FLAGS.merl_dir)):
        brdf = MERL(path=brdf_path)

        # Query the BRDF
        qrusink = brdf.dir2rusink(ldir_rep, vdir) # HxWxLx3
        qrusink_flat = np.reshape(qrusink, (-1, 3))
        rgb_flat = brdf.query(qrusink_flat) # HWLx3

        # Make achromatic
        lumi_flat = xm.img.rgb2lum(rgb_flat)
        lumi = np.reshape(lumi_flat, qrusink.shape[:3]) # HxWxL

        # Integrate
        e_out = np.sum( # L
            lumi * 1. * cos * solid_angles[None, :, None], axis=(0, 1))
        e_out_over_e_in = e_out / (1. * cos[0, 0, :])

        # Plot outgoing energy at each incident direction
        plot_path = join(FLAGS.out_dir, brdf.name + '.png')
        plot = xm.vis.plot.Plot(outpath=plot_path)
        plot.scatter3d(xyz, colors=e_out_over_e_in, equal_axes=True)


if __name__ == '__main__':
    app.run(main)
