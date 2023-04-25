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

from third_party.xiuminglib import xiuminglib as xm
from data_gen.util import save_npz
from brdf.renderer import SphereRenderer
from brdf.merl.merl import MERL


flags.DEFINE_string('indir', '', "directory to downloaded MERL binary files")
flags.DEFINE_float('vali_frac', 0.01, "fraction of data used for validation")
flags.DEFINE_string(
    'envmap_path', 'point', "light probe path or a special string like 'point'")
flags.DEFINE_integer('envmap_h', 16, "light probe height")
flags.DEFINE_float('envmap_inten', 40., "light probe intensity")
flags.DEFINE_float(
    'slice_percentile', 80,
    "clip percentile for visualizing characteristic slice")
flags.DEFINE_integer('ims', 128, "render size during visualization")
flags.DEFINE_integer('spp', 1, "samples per pixel for BRDF rendering")
flags.DEFINE_string('outdir', '', "output directory")
flags.DEFINE_boolean(
    'overwrite', False, "whether to remove output folder if it already exists")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.outdir, rm_if_exists=FLAGS.overwrite)

    brdf = MERL()

    # ------ Testing

    renderer = SphereRenderer(
        FLAGS.envmap_path, FLAGS.outdir, envmap_inten=FLAGS.envmap_inten,
        envmap_h=FLAGS.envmap_h, ims=FLAGS.ims, spp=FLAGS.spp)

    # First 90x90 Rusink. are for the characteristic slice
    cslice_rusink = brdf.get_characterstic_slice_rusink()
    cslice_rusink = np.reshape(cslice_rusink, (-1, 3))

    # Next are for rendering
    render_rusink = brdf.dir2rusink(renderer.ldir, renderer.vdir)
    render_rusink = render_rusink[renderer.lvis.astype(bool)]

    qrusink = np.vstack((cslice_rusink, render_rusink))

    data = {
        'envmap_h': FLAGS.envmap_h, 'ims': FLAGS.ims, 'spp': FLAGS.spp,
        'rusink': qrusink.astype(np.float32)}

    out_path = join(FLAGS.outdir, 'test.npz')
    save_npz(data, out_path)

    # ------ Training & Validation

    brdf_paths = xm.os.sortglob(FLAGS.indir)
    for i, path in enumerate(tqdm(brdf_paths, desc="Training & Validation")):
        if not path.endswith('.binary'):
            continue
            
        brdf = MERL(path=path)

        rusink = brdf.tbl[:, :3]
        refl = brdf.tbl[:, 3:]
        refl = xm.img.rgb2lum(refl)
        refl = refl[:, None]

        # Training-validation split
        n = brdf.tbl.shape[0]
        take_every = int(1 / FLAGS.vali_frac)
        ind = np.arange(0, n)
        vali_ind = np.arange(0, n, take_every, dtype=int)
        train_ind = np.array([x for x in ind if x not in vali_ind])
        train_rusink = rusink[train_ind, :]
        train_refl = refl[train_ind, :]
        vali_rusink = rusink[vali_ind, :]
        vali_refl = refl[vali_ind, :]

        train_data = {
            'i': i, 'name': brdf.name,
            'envmap_h': FLAGS.envmap_h, 'ims': FLAGS.ims, 'spp': FLAGS.spp,
            'rusink': train_rusink.astype(np.float32),
            'refl': train_refl.astype(np.float32)}
        vali_data = {
            'i': i, 'name': brdf.name,
            'envmap_h': FLAGS.envmap_h, 'ims': FLAGS.ims, 'spp': FLAGS.spp,
            'rusink': vali_rusink.astype(np.float32),
            'refl': vali_refl.astype(np.float32)}

        # Dump to disk
        out_path = join(FLAGS.outdir, 'train_%s.npz' % brdf.name)
        save_npz(train_data, out_path)
        out_path = join(FLAGS.outdir, 'vali_%s.npz' % brdf.name)
        save_npz(vali_data, out_path)

        # Visualize
        vis_dir = join(FLAGS.outdir, 'vis')
        for achro in (False, True):
            # Characteristic slice
            cslice = brdf.get_characterstic_slice()
            if achro:
                cslice = xm.img.rgb2lum(cslice)
                cslice = np.tile(cslice[:, :, None], (1, 1, 3))
            cslice_img = brdf.characteristic_slice_as_img(
                cslice, clip_percentile=FLAGS.slice_percentile)
            folder_name = 'cslice'
            if achro:
                folder_name += '_achromatic'
            out_png = join(vis_dir, folder_name, brdf.name + '.png')
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
            out_png = join(vis_dir, folder_name, brdf.name + '.png')
            xm.io.img.write_arr(render, out_png, clip=True)


if __name__ == '__main__':
    app.run(main)
