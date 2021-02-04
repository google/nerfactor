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

# pylint: disable=unsupported-assignment-operation

from os.path import join
import numpy as np
from tqdm import tqdm
from absl import app, flags

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.renderer import SphereRenderer
from google3.experimental.users.xiuming.sim.brdf.merl.merl import MERL


flags.DEFINE_string('merl_dir', '', "")
flags.DEFINE_string('envmap_path', '', "")
flags.DEFINE_integer('envmap_h', 16, "")
flags.DEFINE_float('envmap_inten', 1., "")
flags.DEFINE_float('slice_percentile', 80, "")
flags.DEFINE_integer('ims', 256, "")
flags.DEFINE_integer('spp', 64, "")
flags.DEFINE_string('out_dir', '', "")
flags.DEFINE_boolean('debug', False, "")
flags.DEFINE_boolean('lambert_override', False, "")
flags.DEFINE_boolean('disney_paper_subset', False, "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    renderer = SphereRenderer(
        FLAGS.envmap_path, FLAGS.out_dir, envmap_inten=FLAGS.envmap_inten,
        envmap_h=FLAGS.envmap_h, ims=FLAGS.ims, spp=FLAGS.spp,
        debug=FLAGS.debug)

    for brdf_path in tqdm(xm.os.sortglob(FLAGS.merl_dir)):
        brdf = MERL(path=brdf_path)

        brdf_name = brdf.name
        if FLAGS.disney_paper_subset and brdf_name not in (
                'alumina-oxide', 'light-red-paint', 'orange-paint',
                'gold-paint', 'green-latex', 'yellow-matte-plastic'):
            continue

        for achro in (False, True):
            # Visualize the characteristic slice
            cslice = brdf.get_characterstic_slice()
            if achro:
                cslice = xm.img.rgb2lum(cslice)
                cslice = np.tile(cslice[:, :, None], (1, 1, 3))
            cslice_img = brdf.characteristic_slice_as_img(
                cslice, clip_percentile=FLAGS.slice_percentile)
            folder_name = 'cslice'
            if achro:
                folder_name += '_achromatic'
            out_png = join(FLAGS.out_dir, folder_name, brdf_name + '.png')
            xm.io.img.write_img(cslice_img, out_png)

            # Render with this BRDF
            qrusink = brdf.dir2rusink(renderer.ldir, renderer.vdir)
            lvis = renderer.lvis.astype(bool)
            qrusink_flat = qrusink[lvis]
            rgb_flat = brdf.query(qrusink_flat)
            rgb = np.zeros_like(renderer.lcontrib)
            rgb[lvis] = rgb_flat
            if achro:
                rgb = xm.img.rgb2lum(rgb)
                rgb = np.tile(rgb[:, :, :, None], (1, 1, 1, 3))
            render = renderer.render(rgb)
            folder_name = 'render'
            if achro:
                folder_name += '_achromatic'
            out_png = join(FLAGS.out_dir, folder_name, brdf_name + '.png')
            xm.io.img.write_arr(render, out_png, clip=True)


if __name__ == '__main__':
    app.run(main)
