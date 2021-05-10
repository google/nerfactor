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
from scipy import io as sio
from tqdm import tqdm
from absl import app, flags
import trimesh

from third_party.xiuminglib import xiuminglib as xm


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_string('their_results_dir', '', "directory to their results")
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

    # Load shape
    obj_path = join(FLAGS.scene_dir, 'mesh.obj')
    obj = trimesh.load(obj_path)
    obj = trimesh.ray.ray_pyembree.RayMeshIntersector(obj, scale_to_box=False)

    # ------ Training and validation

    # Load poses
    cams_path = join(FLAGS.scene_dir, 'calib.mat')
    data = sio.loadmat(cams_path)
    poses = data['poses']
    projs = data['projs']

    # Glob image paths
    img_dir = join(FLAGS.scene_dir, 'rgb0')
    img_paths = xm.os.sortglob(img_dir, filename='*', ext='exr')
    assert img_paths, "No image globbed"
    if FLAGS.debug:
        img_paths = img_paths[:4]
        poses = poses[:4, :, :]
        projs = projs[:4, :, :]
    assert len(img_paths) == projs.shape[0], \
        "Numbers of images and camera poses are different"

    # For each view
    imgs, converted_poses = [], []
    factor = None
    for i, img_file in enumerate(tqdm(img_paths, desc="Loading Images")):
        # Load image and render alpha
        img = xm.io.exr.read(img_file)
        alpha = render_alpha(
            obj, poses[i, :, :], projs[i, :, :])
        if factor is None:
            factor = float(img.shape[0]) / FLAGS.h
        else:
            assert float(img.shape[0]) / FLAGS.h == factor, \
                "Images are of varying sizes"
        # Resize
        img = xm.img.resize(img, new_h=FLAGS.h, method='tf')
        alpha = xm.img.resize(alpha[:, :, None], new_h=FLAGS.h, method='tf')
        img = np.dstack((img, alpha))
        imgs.append(img)
        # Corresponding pose
        w2c = poses[i, :, :]
        w2c_rot = w2c[:3, :3]
        w2c_trans = w2c[:3, 3]
        c2w_rot = np.linalg.inv(w2c_rot)
        c2w_trans = -c2w_rot.dot(w2c_trans)
        f = projs[i, 0, 0] / factor
        h_w_f = np.array(img.shape[:2] + (f,))
        pose = np.hstack(( # image width and height are already after resizing
            c2w_rot, c2w_trans.reshape(-1, 1), h_w_f.reshape(-1, 1)))
        converted_poses.append(pose)
    imgs = np.stack(imgs, axis=-1)
    converted_poses = np.dstack(converted_poses)

    # Sanity check
    n_poses = converted_poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, (
        f"Mismatch between numbers of images ({n_imgs}) and "
        f"poses ({n_poses})")

    # Move variable dim to axis 0
    poses = np.moveaxis(converted_poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0)

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

    from IPython import embed; embed()
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


def render_alpha(inter, pose, proj, spp=4):
    obj2cvcam = pose
    cam = xm.camera.PerspCam()
    cam.int_mat = proj
    cam.ext_mat_4x4 = obj2cvcam
    # Cast rays to object
    ray_dirs = cam.gen_rays(spp=spp)
    ray_dirs_flat = np.reshape(ray_dirs, (-1, 3))
    ray_origins = np.tile(cam.loc, (ray_dirs_flat.shape[0], 1))
    hit = inter.intersects_any(ray_origins, ray_dirs_flat)
    hit = np.reshape(hit, (cam.im_h, cam.im_w, spp))
    # Compute alpha
    alpha = np.mean(hit.astype(float), axis=2)
    return alpha


if __name__ == '__main__':
    app.run(main)
