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

import numpy as np

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


def read_img(path, force_white_bg=False, h=None, keep_alpha=False):
    uint_img = xm.io.img.read(path)
    float_img = xm.img.normalize_uint(uint_img)

    if force_white_bg:
        assert float_img.shape[2] == 4, \
            "Can't force white background without alpha channel"
        alpha = float_img[:, :, 3]
        rgb = float_img[:, :, :3]
        bg = np.ones_like(rgb)
        rgb = xm.img.alpha_blend(rgb, alpha, bg)
        float_img = np.dstack((rgb, alpha))

    if (float_img.ndim == 3) and (not keep_alpha):
        float_img = float_img[:, :, :3]

    if h is not None:
        float_img = xm.img.resize(float_img, new_h=h)

    return float_img


def get_colored_cell_str(heat):
    if heat == 0:
        return '& %s '
    if heat == 1:
        return '& \cellcolor{JonYellow}{%s} '
        return '& \cellcolor{red!10!yellow}{%s} '
    if heat == 2:
        return '& \cellcolor{JonOrange}{%s} '
        return '& \cellcolor{red!30!yellow}{%s} '
    if heat == 3:
        return '& \cellcolor{JonRed}{%s} '
        return '& \cellcolor{red!60!yellow}{%s} '
    raise ValueError(heat)


def match_scale(pred, alpha, gt, alpha_thres=0.9, per_ch=False):
    """Scale the prediction (optionally, by each channel) for the best MSE
    with the ground truth.
    """
    is_fg = alpha > alpha_thres
    if per_ch:
        opt_scale = []
        for i in range(3):
            x_hat = pred[:, :, i][is_fg]
            x = gt[:, :, i][is_fg]
            scale = x_hat.dot(x) / x_hat.dot(x_hat)
            opt_scale.append(scale)
        opt_scale = np.array(opt_scale).reshape(1, 1, 3)
    else:
        is_fg = np.dstack([is_fg] * 3)
        x_hat = pred[is_fg]
        x = gt[is_fg]
        opt_scale = x_hat.dot(x) / x_hat.dot(x_hat)
    scaled_pred = opt_scale * pred

    # Composite the scaled prediction (which may have crazy background pixels)
    # onto the GT's background
    pred = xm.img.alpha_blend(scaled_pred, alpha, gt)

    # Clip off the crazy values after scaling
    pred = np.clip(pred, gt.min(), gt.max())

    return pred


def avg_angle(normals1, normals2, alpha, debug=True):
    """Returns degrees.
    """
    normals1 = normals1 * 2 - 1
    normals2 = normals2 * 2 - 1
    normals1_1d = np.reshape(normals1, (-1, 3))
    normals2_1d = np.reshape(normals2, (-1, 3))
    is_fg = alpha == 1
    is_fg = np.ravel(is_fg)
    fg_normals1 = normals1_1d[is_fg]
    fg_normals2 = normals2_1d[is_fg]
    dot = np.sum(fg_normals1 * fg_normals2, axis=1)
    fg_norm1 = np.linalg.norm(fg_normals1, axis=1)
    fg_norm2 = np.linalg.norm(fg_normals2, axis=1)
    norm_prod = fg_norm1 * fg_norm2
    cos = dot / norm_prod
    cos = np.clip(cos, 0, 1) # to safeguard against numerical issues
    theta = np.arccos(cos)
    theta = theta / np.pi * 180 # to degrees
    if debug:
        theta_1d = np.zeros((normals1_1d.shape[0],))
        theta_1d[is_fg] = theta
        theta_2d = np.reshape(theta_1d, normals1.shape[:2])
        xm.vis.matrix.matrix_as_heatmap(theta_2d, outpath='/tmp/theta.png')
    return np.mean(theta)
