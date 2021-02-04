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

# pylint: disable=relative-beyond-top-level,invalid-unary-operand-type

from os.path import basename, dirname, join
import numpy as np
from PIL import Image

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

from .base import Dataset as BaseDataset
from ..util import logging as logutil, io as ioutil, tensor as tutil, \
    img as imgutil


logger = logutil.Logger(loggee="datasets/nerf")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, always_all_rays=False, spp=1):
        self.meta2img = {}
        # To allow supersampling a pixel
        sps = np.sqrt(spp) # samples per side
        assert sps == int(sps), (
            "Samples per pixel must be a square number so that samples per "
            "side are integers")
        self.sps = int(sps)
        # Parent init.
        super().__init__(config, mode, debug=debug)
        # Trigger init. in a main thread before starting multi-threaded work.
        # See http://yaqs/eng/q/6292200559345664 for details
        Image.init()
        # To allow getting all rays for training images
        self.always_all_rays = always_all_rays

    def _get_batch_size(self):
        if self.mode == 'train':
            bs = self.config.getint('DEFAULT', 'n_rays_per_step')
        else:
            # Total number of pixels is batch size, and will need to load
            # a datapoint to figure that out
            any_path = self.files[0]
            ret = self._load_data(any_path)
            map_data = ret[-1] # OK as long as shape is (H, W[, ?])
            bs = int(np.prod(map_data.shape[:2]))
        return bs

    def _glob(self):
        root = self.config.get('DEFAULT', 'data_root')
        if self.mode in ('train', 'test'):
            mode_str = self.mode
        else:
            mode_str = 'val'
        metadata_path_pattern = join(root, '%s_???' % mode_str, 'metadata.json')
        # Shortcircuit if testing
        if self.mode == 'test':
            metadata_paths = sorted(gfile.Glob(metadata_path_pattern))
            logger.info(
                "Number of '%s' views: %d", self.mode, len(metadata_paths))
            return metadata_paths
        # Training or validation
        # Include only cameras with paired RGB images
        metadata_paths = []
        for metadata_path in sorted(gfile.Glob(metadata_path_pattern)):
            img_path = join(dirname(metadata_path), 'rgba.png')
            if gfile.Exists(img_path):
                metadata_paths.append(metadata_path)
                self.meta2img[metadata_path] = img_path
            else:
                logger.warning((
                    "Skipping camera\n\t%s\nbecause its paried RGB image"
                    "\n\t%s\ndoesn't exist"), metadata_path, img_path)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    @staticmethod
    def _parse_id(metadata_path): # pylint: disable=arguments-differ
        return basename(dirname(metadata_path))

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, rayo, rayd, rgb):
        """Records image dimensions and samples rays.
        """
        hw = tf.shape(rgb)[:2]
        rayo, rayd, rgb = self._sample_rays(rayo, rayd, rgb)
        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        return id_, hw, rayo, rayd, rgb

    def _sample_rays(self, rayo, rayd, rgb):
        # Shortcircuit if need all rays
        if self.mode in ('vali', 'test') or self.always_all_rays:
            rayo = tf.reshape(rayo, (-1, 3))
            rayd = tf.reshape(rayd, (-1, 3))
            rgb = tf.reshape(rgb, (-1, 3))
            return rayo, rayd, rgb
        # Training: sample rays
        coords = tf.stack(
            tf.meshgrid(
                tf.range(tf.shape(rgb)[0]), tf.range(tf.shape(rgb)[1]),
                indexing='ij'),
            axis=-1)
        coords = tf.reshape(coords, (-1, 2))
        # Use tf.random instead of np.random here so that the randomness is
        # correct even if we compile this to static graph using tf.function
        select_ind = tf.random.uniform(
            (self.bs,), minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
        select_ind = tf.gather_nd(coords, select_ind[:, None])
        rayo = tf.gather_nd(rayo, select_ind)
        rayd = tf.gather_nd(rayd, select_ind)
        rgb = tf.gather_nd(rgb, select_ind)
        return rayo, rayd, rgb

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, rayo, rayd, rgb = tf.py_function(
            self._load_data, [path],
            (tf.string, tf.float32, tf.float32, tf.float32))
        return id_, rayo, rayd, rgb

    def _load_data(self, metadata_path): # pylint: disable=arguments-differ
        imh = self.config.getint('DEFAULT', 'imh')
        white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)
        # Generate rays
        metadata = ioutil.read_json(metadata_path)
        imw = int(imh / metadata['imh'] * metadata['imw'])
        cam_to_world = np.array([
            float(x) for x in metadata['cam_transform_mat'].split(',')
        ]).reshape(4, 4)
        cam_angle_x = metadata['cam_angle_x']
        rayo, rayd = self._gen_rays(cam_to_world, cam_angle_x, imh, imw)
        rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)
        # Shortcircuit if testing
        if self.mode == 'test':
            rgb = np.zeros((imh, imw, 3), dtype=np.float32) # placeholder
            return id_, rayo, rayd, rgb
        # Training or validation, where each camera has a paired image
        img_path = self.meta2img[metadata_path]
        rgba = xm.io.img.load(img_path)
        assert rgba.ndim == 3 and rgba.shape[2] == 4, "Input image is not RGBA"
        rgba = xm.img.normalize_uint(rgba)
        # Resize RGB
        if imh != rgba.shape[0]:
            rgba = xm.img.resize(rgba, new_h=imh)
        rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
        # Composite RGBA image onto white or black background
        bg = np.ones_like(rgb) if white_bg else np.zeros_like(rgb)
        rgb = imgutil.alpha_blend(rgb, alpha, tensor2=bg)
        rgb = rgb.astype(np.float32)
        return id_, rayo, rayd, rgb

    # pylint: disable=arguments-differ
    def _gen_rays(self, to_world, angle_x, imh, imw):
        near = self.config.getfloat('DEFAULT', 'near')
        ndc = self.config.getboolean('DEFAULT', 'ndc')
        # Ray origin
        cam_loc = to_world[:3, 3]
        rayo = np.tile( # (H * SPS, W * SPS, 3)
            cam_loc[None, None, :], (imh * self.sps, imw * self.sps, 1))
        # Ray directions
        xs = np.linspace(0, imw, imw * self.sps, endpoint=False)
        ys = np.linspace(0, imh, imh * self.sps, endpoint=False)
        xs, ys = np.meshgrid(xs, ys)
        # (0, 0)
        # +--------> (w, 0)
        # |           x
        # |
        # v y (0, h)
        fl = .5 * imw / np.tan(.5 * angle_x)
        rayd = np.stack(
            ((xs - .5 * imw) / fl, -(ys - .5 * imh) / fl, -np.ones_like(xs)),
            axis=-1) # local
        rayd = np.sum(
            rayd[:, :, np.newaxis, :] * to_world[:3, :3], axis=-1) # world
        if ndc:
            # TODO: not in use, so need to check correctness
            # NeRF NDC expects OpenGL coordinates, where up is +y, and forward
            # -z, so we need to flip the rays coming from SfM cameras
            cv2gl_rot = np.diag((1.0, -1.0, -1.0))
            rayo = rayo.dot(cv2gl_rot)
            rayd = rayd.dot(cv2gl_rot)
            # Shift ray origins to near plane
            t = -(near + rayo[..., 2]) / rayd[..., 2]
            rayo += t[..., None] * rayd
            # Projection
            o1 = -1. / (imw / (2. * fl)) * rayo[..., 0] / rayo[..., 2]
            o2 = -1. / (imh / (2. * fl)) * rayo[..., 1] / rayo[..., 2]
            o3 = 1. + 2. * near / rayo[..., 2]
            d1 = -1. / (imw / (2. * fl)) * (
                rayd[..., 0] / rayd[..., 2] - rayo[..., 0] / rayo[..., 2])
            d2 = -1. / (imh / (2. * fl)) * (
                rayd[..., 1] / rayd[..., 2] - rayo[..., 1] / rayo[..., 2])
            d3 = -2. * near / rayo[..., 2]
            rayo = np.dstack((o1, o2, o3))
            rayd = np.dstack((d1, d2, d3))
        return rayo, rayd
