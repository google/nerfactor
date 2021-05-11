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
from data_gen.util import read_bundle_file, recenter_poses, spherify_poses


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_integer('h', 512, "output image height")
flags.DEFINE_integer('n_vali', 2, "number of held-out validation views")
flags.DEFINE_string('exclude', '', "indices of frames to exclude")
flags.DEFINE_string('outroot', '', "output root")
flags.DEFINE_boolean('debug', False, "debug toggle")
FLAGS = flags.FLAGS


def main(_):
    view_folder = '{mode}_{i:03d}'

    # Only the original NeRF and JaxNeRF implementations need these
    train_json = join(FLAGS.outroot, 'transforms_train.json')
    vali_json = join(FLAGS.outroot, 'transforms_val.json')
    test_json = join(FLAGS.outroot, 'transforms_test.json')

    # ------ Training and validation

    if FLAGS.exclude == '':
        exclude = []
    else:
        exclude = [int(x) for x in FLAGS.exclude.split(',')]

    # Load poses
    bundle_path = join(FLAGS.scene_dir, 'cameras', 'bundle.out')
    cams, _ = read_bundle_file(bundle_path)
    cams = [x for i, x in enumerate(cams) if i not in exclude]

    # Load and resize images, and then convert their corresponding poses
    img_dir = join(FLAGS.scene_dir, 'images')
    img_paths = xm.os.sortglob(
        img_dir, filename='*', ext=('jpg', 'jpeg', 'png'), ext_ignore_case=True)
    img_paths = [x for i, x in enumerate(img_paths) if i not in exclude]
    assert img_paths, "No image globbed"
    if FLAGS.debug:
        img_paths = img_paths[:4]
        cams = cams[:4]
    imgs, poses = [], []
    factor = None
    for i, img_file in enumerate(tqdm(img_paths, desc="Loading images")):
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
        # Corresponding pose
        cam = cams[i]
        w2c_rot = cam['R']
        w2c_trans = cam['T']
        c2w_rot = np.linalg.inv(w2c_rot)
        c2w_trans = -c2w_rot.dot(w2c_trans)
        h_w_f = np.array(img.shape[:2] + (cam['f'],))
        pose = np.hstack(( # image width and height are already after resizing
            c2w_rot, c2w_trans.reshape(-1, 1), h_w_f.reshape(-1, 1)))
        poses.append(pose)
    imgs = np.stack(imgs, axis=-1)
    poses = np.dstack(poses)

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, (
        f"Mismatch between numbers of images ({n_imgs}) and "
        f"poses ({n_poses})")

    # Update focal length according to downsampling
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Move variable dim to axis 0
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0)

    # Recenter poses
    poses = recenter_poses(poses)

    # Generate a spiral/spherical path for rendering videos
    poses, test_poses = spherify_poses(poses)

    # Training-validation split
    ind_vali = np.arange(n_imgs)[:-1:(n_imgs // FLAGS.n_vali)]
    ind_train = np.array(
        [x for x in np.arange(n_imgs) if x not in ind_vali])

    # Figure out camera angle
    fl = poses[0, -1, -1]
    cam_angle_x = np.arctan2(imgs.shape[2] / 2, fl) * 2

    # Training frames
    train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_train):
        view_folder_ = view_folder.format(mode='train', i=vi)
        # Write image
        img = imgs[i, :, :, :]
        xm.io.img.write_float(
            img, join(FLAGS.outroot, view_folder_, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {
            'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
            'transform_matrix': c2w.tolist()}
        train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        xm.io.json.write(
            frame_meta, join(FLAGS.outroot, view_folder_, 'metadata.json'))

    # Validation views
    vali_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_vali):
        view_folder_ = view_folder.format(mode='val', i=vi)
        # Write image
        img = imgs[i, :, :, :]
        xm.io.img.write_float(
            img, join(FLAGS.outroot, view_folder_, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {
            'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
            'transform_matrix': c2w.tolist()}
        vali_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        xm.io.json.write(
            frame_meta, join(FLAGS.outroot, view_folder_, 'metadata.json'))

    # Write training and validation JSONs
    xm.io.json.write(train_meta, train_json)
    xm.io.json.write(vali_meta, vali_json)

    # ------ Testing

    # Test views
    test_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for i in range(test_poses.shape[0]):
        view_folder_ = view_folder.format(mode='test', i=i)
        # Record metadata
        pose = test_poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {
            'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
        test_meta['frames'].append(frame_meta)
        # Write the nearest input to this test view folder
        dist = np.linalg.norm(pose[:, 3] - poses[:, :, 3], axis=1)
        nn_i = np.argmin(dist)
        nn_img = imgs[nn_i, :, :, :]
        xm.io.img.write_float(
            nn_img, join(FLAGS.outroot, view_folder_, 'nn.png'), clip=True)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0, 'original_path': ''}
        xm.io.json.write(
            frame_meta, join(FLAGS.outroot, view_folder_, 'metadata.json'))

    # Write JSON
    xm.io.json.write(test_meta, test_json)


if __name__ == '__main__':
    app.run(main)
