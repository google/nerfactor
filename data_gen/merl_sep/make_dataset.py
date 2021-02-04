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

from os.path import join, basename
import numpy as np
from tqdm import tqdm
from absl import app, flags

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.data_gen.util import save_npz
from google3.experimental.users.xiuming.sim.brdf.renderer import SphereRenderer
from google3.experimental.users.xiuming.sim.brdf.merl.merl import MERL
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil


flags.DEFINE_string('merl_dir', '', "")
flags.DEFINE_string('merl_sep_dir', '', "")
flags.DEFINE_float('vali_frac', 0.01, "")
flags.DEFINE_string('envmap_path', 'point', "")
flags.DEFINE_integer('envmap_h', 16, "")
flags.DEFINE_float('envmap_inten', 40, "")
flags.DEFINE_float('slice_percentile', 80, "")
flags.DEFINE_integer('ims', 256, "")
flags.DEFINE_integer('spp', 1, "")
flags.DEFINE_string('out_dir', '', "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.out_dir, rm_if_exists=True)

    # Initialize a Lambertian BRDF
    brdf = MERL()

    renderer = SphereRenderer(
        FLAGS.envmap_path, FLAGS.out_dir, envmap_inten=FLAGS.envmap_inten,
        envmap_h=FLAGS.envmap_h, ims=FLAGS.ims, spp=FLAGS.spp)

    # ------ Testing

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

    # Mask of valid bins in MERL
    mask_npy = join(FLAGS.merl_sep_dir, 'maskMap.npy')
    mask = ioutil.load_np(mask_npy) # number of True's: M
    mask = np.reshape(mask, brdf.cube_rgb.shape[:3]) # 180x90x90

    # Specular lobes
    spec_npy = join(FLAGS.merl_sep_dir, 'specAll_log2.npy')
    spec_all = ioutil.load_np(spec_npy) # MxN

    # Glob the BRDF names
    merl_binaries = xm.os.sortglob(FLAGS.merl_dir, '*.binary')
    assert len(merl_binaries) == 100, \
        "There should be exactly 100 MERL materials"
    brdf_names = [basename(x)[:-len('.binary')] for x in merl_binaries]

    for i, name in enumerate(tqdm(brdf_names, desc="Training & Validation")):
        # Set reflectance
        spec = -np.ones(mask.shape, dtype=spec_all.dtype) # 180x90x90
        spec[mask] = spec_all[:, i]
        spec = np.stack([spec] * 3, axis=-1)
        brdf.cube_rgb = spec

        # Visualize the characteristic slice
        cslice = brdf.get_characterstic_slice()
        cslice_img = brdf.characteristic_slice_as_img(
            cslice, clip_percentile=FLAGS.slice_percentile)
        out_png = join(FLAGS.out_dir, 'vis-cslice_%s.png' % name)
        xm.io.img.write_img(cslice_img, out_png)

        # Render with this BRDF
        qrusink = brdf.dir2rusink(renderer.ldir, renderer.vdir)
        lvis = renderer.lvis.astype(bool)
        qrusink_flat = qrusink[lvis]
        rgb_flat = brdf.query(qrusink_flat)
        rgb = np.zeros_like(renderer.lcontrib)
        rgb[lvis] = rgb_flat
        render = renderer.render(rgb)
        out_png = join(FLAGS.out_dir, 'vis-render_%s.png' % name)
        xm.io.img.write_arr(render, out_png, clip=True)

        rusink = brdf.tbl[:, :3]
        refl = brdf.tbl[:, 3:4] # Nx1

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
            'i': i, 'name': name,
            'envmap_h': FLAGS.envmap_h, 'ims': FLAGS.ims, 'spp': FLAGS.spp,
            'rusink': train_rusink.astype(np.float32),
            'refl': train_refl.astype(np.float32)}
        vali_data = {
            'i': i, 'name': name,
            'envmap_h': FLAGS.envmap_h, 'ims': FLAGS.ims, 'spp': FLAGS.spp,
            'rusink': vali_rusink.astype(np.float32),
            'refl': vali_refl.astype(np.float32)}

        # Dump to disk
        out_path = join(FLAGS.out_dir, 'train_%s.npz' % name)
        save_npz(train_data, out_path)
        out_path = join(FLAGS.out_dir, 'vali_%s.npz' % name)
        save_npz(vali_data, out_path)


if __name__ == '__main__':
    app.run(main)
