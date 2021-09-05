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

# pylint: disable=invalid-unary-operand-type

from os.path import join
import numpy as np

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil
from nerfactor.datasets.nerf_shape import Dataset as BaseDataset


logger = logutil.Logger(loggee="datasets/mvs_shape")


class Dataset(BaseDataset):
    def _glob(self):
        mvs_root = self.config.get('DEFAULT', 'mvs_root')
        # Glob metadata paths
        mode_str = 'val' if self.mode == 'vali' else self.mode
        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(mvs_root, '%s_000' % mode_str)
        else:
            metadata_dir = join(mvs_root, '%s_???' % mode_str)
        # Include only cameras with all required buffers (depending on mode)
        metadata_paths, incomplete_paths = [], []
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            id_ = self._parse_id(metadata_path)
            view_dir = join(mvs_root, id_)
            lvis_path = join(view_dir, 'lvis.npy')
            normal_path = join(view_dir, 'normal.npy')
            xyz_path = join(view_dir, 'xyz.npy')
            alpha_path = join(view_dir, 'alpha.png')
            paths = {
                'xyz': xyz_path, 'normal': normal_path, 'lvis': lvis_path,
                'alpha': alpha_path}
            if self.mode != 'test':
                rgba_path = join(view_dir, 'rgba.png')
                paths['rgba'] = rgba_path
            if ioutil.all_exist(paths):
                metadata_paths.append(metadata_path)
                self.meta2buf[metadata_path] = paths
            else:
                incomplete_paths.append(metadata_path)
        if incomplete_paths:
            logger.warn((
                "Skipping\n\t%s\nbecause at least one of their paired "
                "buffers doesn't exist"), incomplete_paths)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    # pylint: disable=arguments-differ
    def _load_data(self, metadata_path):
        imh = self.config.getint('DEFAULT', 'imh')
        use_nerf_alpha = self.config.getboolean('DEFAULT', 'use_nerf_alpha')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)
        # Rays
        metadata = ioutil.read_json(metadata_path)
        h, w = metadata['imh'], metadata['imw']
        cam_loc = np.array(metadata['cam_loc'])
        rayo = np.tile(cam_loc[None, None, :], (h, w, 1))
        rayo = rayo.astype(np.float32)
        rayd = np.zeros_like(rayo) # dummy
        # Load precomputed shape properties
        paths = self.meta2buf[metadata_path]
        xyz = ioutil.load_np(paths['xyz'])
        normal = ioutil.load_np(paths['normal'])
        if self.debug:
            logger.warn("Faking light visibility for faster debugging")
            lvis = 0.5 * np.ones(normal.shape[:2] + (512,), dtype=np.float32)
        else:
            lvis = ioutil.load_np(paths['lvis'])
        # RGB and alpha, depending on the mode
        if self.mode == 'test':
            # No RGBA, so estimated alpha and placeholder RGB
            alpha = xm.io.img.load(paths['alpha'])
            alpha = xm.img.normalize_uint(alpha)
            rgb = np.zeros_like(xyz)
        else:
            # Training or validation, where each camera has a paired image
            rgba = xm.io.img.load(paths['rgba'])
            assert rgba.ndim == 3 and rgba.shape[2] == 4, \
                "Input image is not RGBA"
            rgba = xm.img.normalize_uint(rgba)
            rgb = rgba[:, :, :3]
            if use_nerf_alpha: # useful for real scenes
                alpha = xm.io.img.load(paths['alpha'])
                alpha = xm.img.normalize_uint(alpha)
            else:
                alpha = rgba[:, :, 3] # ground-truth alpha
        # Resize
        if imh != xyz.shape[0]:
            xyz = xm.img.resize(xyz, new_h=imh)
            normal = xm.img.resize(normal, new_h=imh)
            lvis = xm.img.resize(lvis, new_h=imh)
            alpha = xm.img.resize(alpha, new_h=imh)
            rgb = xm.img.resize(rgb, new_h=imh)
        # Make sure there's no XYZ coinciding with camera (caused by occupancy
        # accumulating to 0)
        assert not np.isclose(xyz, rayo).all(axis=2).any(), \
            "Found XYZs coinciding with the camera"
        # Re-normalize normals and clip light visibility before returning
        normal = xm.linalg.normalize(normal, axis=2)
        assert np.isclose(np.linalg.norm(normal, axis=2), 1).all(), \
            "Found normals with a norm far away from 1"
        lvis = np.clip(lvis, 0, 1)
        return id_, rayo, rayd, rgb, alpha, xyz, normal, lvis
