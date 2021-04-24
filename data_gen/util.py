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
import numpy as np


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
            break # TODO some file has a different number of points than what's specified in the header
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
