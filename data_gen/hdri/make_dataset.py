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

from google3.pyglib import gfile
from google3.experimental.users.xiuming.sim.brdf.renderer import gen_light_xyz
from google3.experimental.users.xiuming.sim.data_gen.util import save_npz
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string('hdri_dir', '', "")
flags.DEFINE_integer('start_i', 0, "")
flags.DEFINE_float('vali_frac', 0.01, "")
flags.DEFINE_integer('envmap_h', 256, "")
flags.DEFINE_string('outdir', '', "")
FLAGS = flags.FLAGS


def main(_):
    # Keep only outdoor maps
    paths = []
    for path in gfile.Glob(join(FLAGS.hdri_dir, '*.hdr')):
        i = basename(path)[:-len('.hdr')]
        i = int(i)
        if i >= FLAGS.start_i:
            paths.append(path)
    paths = sorted(paths)

    xm.os.makedirs(FLAGS.outdir, rm_if_exists=True)

    # ------ Training & Validation

    for i, path in enumerate(tqdm(paths, desc="Training & Validation")):
        id_ = basename(path)[:-len('.hdr')]

        # Load HDR data
        hdr = xm.io.hdr.read(path)
        hdr_down = xm.img.resize(hdr, new_h=FLAGS.envmap_h, use_tf=True)

        # Visualize both the original and downsampled resolutions
        outpath = join(FLAGS.outdir, 'vis_' + id_ + '_orig.png')
        vis_hdr(hdr, outpath)
        outpath = join(FLAGS.outdir, 'vis_' + id_ + '.png')
        vis_hdr(hdr_down, outpath)

        # Generate latitudes and longitudes for these light pixels
        xyz, _ = gen_light_xyz(hdr_down.shape[0], hdr_down.shape[1])
        xyz_flat = np.reshape(xyz, (-1, 3))
        rlatlng = xm.geometry.sph.cart2sph(xyz_flat)
        latlng_flat = rlatlng[:, 1:]

        # Training-validation split
        n = latlng_flat.shape[0]
        take_every = int(1 / FLAGS.vali_frac)
        ind = np.arange(0, n)
        vali_ind = np.arange(0, n, take_every, dtype=int)
        train_ind = np.array([x for x in ind if x not in vali_ind])
        train_latlng = latlng_flat[train_ind, :]
        hdr_flat = np.reshape(hdr_down, (-1, 3))
        train_radi = hdr_flat[train_ind, :]
        vali_latlng = latlng_flat[vali_ind, :]
        vali_radi = hdr_flat[vali_ind, :]

        train_data = {
            'i': i, 'name': id_, 'envmap_h': hdr_down.shape[0],
            'latlng': train_latlng.astype(np.float32),
            'radi': train_radi.astype(np.float32)}
        vali_data = {
            'i': i, 'name': id_, 'envmap_h': hdr_down.shape[0],
            'latlng': vali_latlng.astype(np.float32),
            'radi': vali_radi.astype(np.float32)}

        # Dump to disk
        out_path = join(FLAGS.outdir, 'train_%s.npz' % id_)
        save_npz(train_data, out_path)
        out_path = join(FLAGS.outdir, 'vali_%s.npz' % id_)
        save_npz(vali_data, out_path)

    # ------ Testing

    # Generate latitudes and longitudes for these light pixels
    xyz, _ = gen_light_xyz(FLAGS.envmap_h, 2 * FLAGS.envmap_h)
    xyz_flat = np.reshape(xyz, (-1, 3))
    rlatlng = xm.geometry.sph.cart2sph(xyz_flat)
    latlng_flat = rlatlng[:, 1:]

    data = {
        'envmap_h': FLAGS.envmap_h, 'latlng': latlng_flat.astype(np.float32)}

    out_path = join(FLAGS.outdir, 'test.npz')
    save_npz(data, out_path)


def vis_hdr(hdr, outpath):
    tonemapped = xm.img.tonemap(hdr) # [0, 1]
    linear = tonemapped / tonemapped.max() # should be redundant
    srgb = xm.img.linear2srgb(linear)
    xm.io.img.write_arr(srgb, outpath)


if __name__ == '__main__':
    app.run(main)
