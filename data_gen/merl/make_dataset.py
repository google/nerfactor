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


flags.DEFINE_string('merl_dir', '', "directory to downloaded MERL binary files")
flags.DEFINE_float('vali_frac', 0.01, "fraction of data used for validation")
flags.DEFINE_string(
    'envmap_path', 'point', "light probe path or a special string like 'point'")
flags.DEFINE_integer('envmap_h', 16, "light probe height")
flags.DEFINE_float('envmap_inten', 1., "light probe intensity")
flags.DEFINE_integer('ims', 256, "render size during visualization")
flags.DEFINE_integer('spp', 64, "samples per pixel for BRDF rendering")
flags.DEFINE_string('out_dir', '', "output directory")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    brdf = MERL()

    # ------ Testing

    renderer = SphereRenderer(
        FLAGS.envmap_path, FLAGS.out_dir, envmap_inten=FLAGS.envmap_inten,
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

    out_path = join(FLAGS.out_dir, 'test.npz')
    save_npz(data, out_path)

    # ------ Training & Validation

    brdf_paths = xm.os.sortglob(FLAGS.merl_dir)
    for i, path in enumerate(tqdm(brdf_paths, desc="Training & Validation")):
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
        out_path = join(FLAGS.out_dir, 'train_%s.npz' % brdf.name)
        save_npz(train_data, out_path)
        out_path = join(FLAGS.out_dir, 'vali_%s.npz' % brdf.name)
        save_npz(vali_data, out_path)


if __name__ == '__main__':
    app.run(main)
