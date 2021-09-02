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
from data_gen.util import gen_data


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_integer('h', 512, "output image height")
flags.DEFINE_integer('n_vali', 2, "number of held-out validation views")
flags.DEFINE_float('bound_factor', .75, "bound factor")
flags.DEFINE_string('outroot', '', "output root")
flags.DEFINE_boolean('debug', False, "debug toggle")
FLAGS = flags.FLAGS


def main(_):
    # Load poses
    poses_path = join(FLAGS.scene_dir, 'poses_bounds.npy')
    poses_arr = xm.io.np.read_or_write(poses_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Load and resize images
    img_dir = join(FLAGS.scene_dir, 'images')
    img_paths = xm.os.sortglob(
        img_dir, filename='*', ext='jpg', ext_ignore_case=True)
    assert img_paths, "No image globbed"
    if FLAGS.debug:
        img_paths = img_paths[:4]
        poses = poses[..., :4]
        bds = bds[..., :4]
    imgs = []
    factor = None
    for img_file in tqdm(img_paths, desc="Loading images"):
        img = xm.io.img.read(img_file)
        img = xm.img.normalize_uint(img)
        if factor is None:
            factor = float(img.shape[0]) / FLAGS.h
        else:
            assert float(img.shape[0]) / FLAGS.h == factor, \
                "Images are of varying sizes"
        img = xm.img.resize(img, new_h=FLAGS.h, method='tf')
        if img.shape[2] == 3:
            # NOTE: add an all-one alpha
            img = np.dstack((img, np.ones_like(img)[:, :, :1]))
        imgs.append(img)
    imgs = np.stack(imgs, axis=-1)

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, (
        "Mismatch between numbers of images ({n_imgs}) and "
        "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)

    # Update poses according to downsampling
    poses[:2, 4, :] = np.array(
        imgs.shape[:2]).reshape([2, 1]) # override image size
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor # scale focal length

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # Nx3x5
    imgs = np.moveaxis(imgs, -1, 0) # NxHxWx4
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # Nx2

    # Rescale according to a default bd factor
    scale = 1. / (bds.min() * FLAGS.bound_factor)
    poses[:, :3, 3] *= scale # scale translation
    bds *= scale

    gen_data(poses, imgs, img_paths, FLAGS.n_vali, FLAGS.outroot)


if __name__ == '__main__':
    app.run(main)
