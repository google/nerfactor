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


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_integer('h', 512, "output image height")
flags.DEFINE_integer('n_vali', 2, "number of held-out validation views")
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
    imgs = np.moveaxis(imgs, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # Nx2

    # Rescale according to a default bd factor
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale # scale translation
    bds *= scale

    # Recenter poses
    poses = _recenter_poses(poses)

    # Generate a spiral/spherical ray path for rendering videos
    poses, test_poses = _generate_spherical_poses(poses, bds)

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


def _recenter_poses(poses):
    """Recenter poses according to the original NeRF code.
    """
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = _poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def _poses_avg(poses):
    """Average poses according to the original NeRF code.
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = _normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def _normalize(x):
    """Normalization helper function.
    """
    return x / np.linalg.norm(x)


def _viewmatrix(z, up, pos):
    """Construct lookat view matrix.
    """
    vec2 = _normalize(z)
    vec1_avg = up
    vec0 = _normalize(np.cross(vec1_avg, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def _generate_spherical_poses(poses, bds):
    """Generate a 360 degree spherical path for rendering.
    """
    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -a_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv(
            (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = _normalize(up)
    vec1 = _normalize(np.cross([.1, .2, .3], vec0))
    vec2 = _normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([
            radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        vec2 = _normalize(camorigin)
        vec0 = _normalize(np.cross(vec2, up))
        vec1 = _normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    render_poses = new_poses[:, :3, :4]
    return poses_reset, render_poses


if __name__ == '__main__':
    app.run(main)
