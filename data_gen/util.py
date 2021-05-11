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

from io import BytesIO
from os.path import basename
import numpy as np

from third_party.xiuminglib import xiuminglib as xm


def spherify_poses(poses):
    """poses: Nx3x5 (final column contains H, W, and focal length)."""
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
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)

    new_poses = []
    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([
            radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
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
    return poses_reset, new_poses


def recenter_poses(poses):
    """Recenter poses according to the original NeRF code.
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
    return poses


def poses_avg(poses):
    """Average poses according to the original NeRF code.
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def normalize(x):
    """Normalization helper function.
    """
    return x / np.linalg.norm(x)


def _viewmatrix(z, up, pos):
    """Construct lookat view matrix.
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def read_bundle_file(path):
    """Reads cameras and points from a bundle file (format:
    https://github.com/snavely/bundler_sfm#output-format).
    """
    with open(path, 'r') as h:
        lines = list(h)
    lines = [x.rstrip() for x in lines]

    n_cam, n_pts = lines[1].split(' ')
    n_cam, n_pts = int(n_cam), int(n_pts)

    # Cameras
    cams = []
    for i in range(n_cam):
        j = 2 + 5 * i
        f_k1_k2 = lines[j]
        rot_row1 = lines[j + 1]
        rot_row2 = lines[j + 2]
        rot_row3 = lines[j + 3]
        trans = lines[j + 4]
        f = float(f_k1_k2.split(' ')[0])
        rot = np.vstack((
            [float(x) for x in rot_row1.split(' ')],
            [float(x) for x in rot_row2.split(' ')],
            [float(x) for x in rot_row3.split(' ')]))
        trans = np.array([float(x) for x in trans.split(' ')])
        cam = {'f': f, 'R': rot, 'T': trans}
        cams.append(cam)
    assert len(cams) == n_cam, (
        "A different number of cameras read than what is specified in the "
        "header")

    # Points
    pts = []
    for i in range(n_pts):
        j = 2 + 5 * n_cam + 3 * i
        if j == len(lines):
            print("# points different than what's specified in the header")
            break
        xyz = lines[j]
        rgb = lines[j + 1]
        views = lines[j + 2]
        xyz = np.array([float(x) for x in xyz.split(' ')])
        rgb = np.array([int(x) for x in rgb.split(' ')])
        views = views.split(' ')
        imgs = []
        for vi in range(int(views[0])):
            k = 1 + 4 * vi
            cam_i = int(views[k])
            kpt_i = int(views[k + 1])
            xy = np.array([float(views[k + 2]), float(views[k + 3])])
            img = {'cam_i': cam_i, 'kpt_i': kpt_i, 'xy': xy}
            imgs.append(img)
        pt = {'xyz': xyz, 'rgb': rgb, 'imgs': imgs}
        pts.append(pt)

    return cams, pts


def save_npz(dict_, path):
    """The extra hassle is for Google infra.
    """
    with open(path, 'wb') as h:
        io_buffer = BytesIO()
        np.savez(io_buffer, **dict_)
        h.write(io_buffer.getvalue())


def read_light(path):
    ext = basename(path).split('.')[-1]
    if ext == 'exr':
        arr = xm.io.exr.read(path)
    elif ext == 'hdr':
        arr = xm.io.hdr.read(path)
    else:
        raise NotImplementedError(ext)
    return arr


def listify_matrix(mat):
    elements = []
    for row in mat:
        for x in row:
            elements.append(x)
    return elements
