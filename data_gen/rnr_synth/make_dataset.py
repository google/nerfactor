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
from data_gen.util import poses_avg, normalize


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_string(
    'their_qual_results_dir', '',
    "directory to their results for qualitative comparisons")
flags.DEFINE_string(
    'their_quan_results_dir', '',
    "directory to their results for quantitative comparisons")
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

    cvcam2glcam = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # For each training or validation view
    imgs, converted_poses = [], []
    factor = None
    for i, img_file in enumerate(tqdm(img_paths, desc="Loading Images")):
        # Load image and render alpha
        img = xm.io.exr.read(img_file)
        alpha = render_alpha(obj, poses[i, :, :], projs[i, :, :])
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
        w2cvcam = poses[i, :3, :] # 3x4
        w2glcam = cvcam2glcam.dot(w2cvcam)
        w2c_rot = w2glcam[:3, :3]
        w2c_trans = w2glcam[:3, 3]
        c2w_rot = np.linalg.inv(w2c_rot)
        c2w_trans = -c2w_rot.dot(w2c_trans)
        f = projs[i, 0, 0] / factor # adjust focal length
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
    converted_poses = np.moveaxis(converted_poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0)

    # Load test poses
    cams_path = join(FLAGS.scene_dir, 'test_seq', 'spiral_step720', 'calib.mat')
    data = sio.loadmat(cams_path)
    test_poses = data['poses'] # w2cvcam
    converted_test_poses = []
    for i in range(test_poses.shape[0]):
        w2cvcam = test_poses[i, :, :] # 4x4
        w2glcam = cvcam2glcam.dot(w2cvcam[:3, :]) # 3x4
        w2c = np.vstack((w2glcam, [0, 0, 0, 1])) # 4x4
        c2w = np.linalg.inv(w2c) # 4x4
        pose = np.hstack((c2w[:3, :], np.reshape(h_w_f, (-1, 1))))
        converted_test_poses.append(pose)
    converted_test_poses = np.stack(converted_test_poses, axis=0)

    # Recenter poses
    converted_poses, converted_test_poses = recenter_poses(
        converted_poses, converted_test_poses)

    # Generate a spiral/spherical path for rendering videos
    converted_poses, converted_test_poses = spherify_poses(
        converted_poses, converted_test_poses)

    # Training-validation split
    ind_vali = np.arange(n_imgs)[:-1:(n_imgs // FLAGS.n_vali)]
    ind_train = np.array(
        [x for x in np.arange(n_imgs) if x not in ind_vali])

    # Figure out camera angle
    cam_angle_x = None
    for i in range(converted_poses.shape[0]):
        fl = converted_poses[i, -1, -1]
        if cam_angle_x is None:
            cam_angle_x = np.arctan2(imgs.shape[2] / 2, fl) * 2
        else:
            assert cam_angle_x == np.arctan2(imgs.shape[2] / 2, fl) * 2, \
                "The frames have different focal lengths"

    # Training frames
    train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_train):
        view_folder_ = view_folder.format(mode='train', i=vi)
        # Write image
        img = imgs[i, :, :, :]
        xm.io.img.write_float(
            img, join(FLAGS.outroot, view_folder_, 'rgba.png'), clip=True)
        # Record metadata
        pose = converted_poses[i, :, :]
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
        pose = converted_poses[i, :, :]
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

    test_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    test_i = 0

    # Their quantitative results and alpha
    alpha_paths = xm.os.sortglob(
        join(FLAGS.their_quan_results_dir, 'alpha_map'), '*', ext='png')
    pred_paths = xm.os.sortglob(
        join(FLAGS.their_quan_results_dir, 'img_est_001'), '*', ext='png')
    gt_paths = xm.os.sortglob(
        join(FLAGS.their_quan_results_dir, 'img_gt_001'), '*', ext='png')

    # For each quantitative test view
    for i, alpha_path in enumerate(
            tqdm(alpha_paths, desc="Quantitative Views")):
        view_folder_ = view_folder.format(mode='test', i=test_i)
        c2w = converted_poses[i * 10, :, :] # 4x4
        # Record metadata
        frame_meta = {
            'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
        test_meta['frames'].append(frame_meta)
        # Write the nearest input to this test view folder
        dist = np.linalg.norm(c2w[:3, 3] - converted_poses[:, :3, 3], axis=1)
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
        # Also copy over their results
        alpha = xm.io.img.read(alpha_path)
        write_their_result(
            pred_paths[i], alpha,
            join(FLAGS.outroot, view_folder_, 'lp1_rnr.png'))
        write_their_result(
            gt_paths[i], alpha, join(FLAGS.outroot, view_folder_, 'lp1_gt.png'))
        test_i += 1

    # Their qualitative results and alpha
    alpha_paths = xm.os.sortglob(
        join(FLAGS.their_qual_results_dir, 'alpha_map'), '*', ext='png')
    lp0_paths = xm.os.sortglob(
        join(FLAGS.their_qual_results_dir, 'img_est_000'), '*', ext='png')
    lp1_paths = xm.os.sortglob(
        join(FLAGS.their_qual_results_dir, 'img_est_001'), '*', ext='png')
    lp2_paths = xm.os.sortglob(
        join(FLAGS.their_qual_results_dir, 'img_est_002'), '*', ext='png')
    lp3_paths = xm.os.sortglob(
        join(FLAGS.their_qual_results_dir, 'img_est_003'), '*', ext='png')

    # For each qualitative test view
    for i in tqdm(
            range(converted_test_poses.shape[0]), desc="Qualitative Views"):
        view_folder_ = view_folder.format(mode='test', i=test_i)
        c2w = converted_test_poses[i, :, :] # 4x4
        # Record metadata
        frame_meta = {
            'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
        test_meta['frames'].append(frame_meta)
        # Write the nearest input to this test view folder
        dist = np.linalg.norm(c2w[:3, 3] - converted_poses[:, :3, 3], axis=1)
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
        # Also copy over their results
        alpha = xm.io.img.read(alpha_paths[i])
        write_their_result(
            lp0_paths[i], alpha, join(FLAGS.outroot, view_folder_, 'lp0.png'))
        write_their_result(
            lp1_paths[i], alpha, join(FLAGS.outroot, view_folder_, 'lp1.png'))
        write_their_result(
            lp2_paths[i], alpha, join(FLAGS.outroot, view_folder_, 'lp2.png'))
        write_their_result(
            lp3_paths[i], alpha, join(FLAGS.outroot, view_folder_, 'lp3.png'))
        test_i += 1

    # Write JSON
    xm.io.json.write(test_meta, test_json)


def recenter_poses(poses, test_poses):
    """Processes the test poses the same way as training or validation poses.
    """
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    # Do the same to test poses
    test_poses_ = test_poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [test_poses.shape[0], 1, 1])
    test_poses = np.concatenate([test_poses[:, :3, :4], bottom], -2)
    test_poses = np.linalg.inv(c2w) @ test_poses
    test_poses_[:, :3, :4] = test_poses[:, :3, :4]
    test_poses = test_poses_

    return poses, test_poses


def spherify_poses(poses, test_poses):
    """Processes the test poses the same way as training or validation poses.
    """
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4] # because pose is camera-to-world

    def p34_to_44(p):
        """p: Nx3x4."""
        return np.concatenate((
            p,
            np.tile(
                np.reshape(np.eye(4)[-1, :], (1, 1, 4)),
                (p.shape[0], 1, 1)),
        ), 1)

    def min_line_dist(rays_o, rays_d):
        a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -a_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv(
            (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)

    # Do the same to test poses
    test_poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(test_poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(test_poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    test_poses_reset[:, :3, 3] *= sc
    test_poses_reset = np.concatenate([
        test_poses_reset[:, :3, :4],
        np.broadcast_to(
            test_poses[0, :3, -1:], test_poses_reset[:, :3, -1:].shape)
    ], -1)

    return poses_reset, test_poses_reset


def write_their_result(inpath, alpha, outpath):
    im = xm.io.img.read(inpath)
    im_rgba = np.concatenate((im, alpha[:, :, None]), axis=2)
    assert im_rgba.shape[0] == FLAGS.h, \
        "Their result has a different resolution"
    xm.io.img.write_uint(im_rgba, outpath)


def render_alpha(inter, pose, proj, spp=4):
    # Set camera
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
