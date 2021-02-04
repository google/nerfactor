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

# pylint: disable=relative-beyond-top-level

from os.path import basename, join
import numpy as np
from PIL import Image

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

from .base import Dataset as BaseDataset
from ..util import logging as logutil, cam as camutil, io as ioutil


logger = logutil.Logger(loggee="datasets/nerf_sfm_cam")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, always_all_rays=False):
        self.campath2imgpath = {}
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
            rgb = ret[-1]
            bs = int(np.prod(rgb.shape[:2]))
        return bs

    def _glob(self):
        root = self.config.get('DEFAULT', 'data_root')
        holdout = self.config.get('DEFAULT', 'holdout')
        holdout = holdout.split(',')
        trainvali_cam_dir = join(root, 'trainvali_cam')
        trainvali_img_dir = join(root, 'trainvali_rgb')
        test_cam_dir = join(root, 'test_cam')
        # Handles different naming conventions by others
        if not gfile.Exists(trainvali_cam_dir):
            trainvali_cam_dir = join(root, 'camera')
        if not gfile.Exists(trainvali_img_dir):
            trainvali_img_dir = join(root, 'rgb')
        if not gfile.Exists(test_cam_dir):
            test_cam_dir = join(root, 'test_camera')
        # Shortcircuit if testing
        if self.mode == 'test':
            cam_paths = sorted(gfile.Glob(join(test_cam_dir, '*.pb')))
            logger.info("Number of %s views: %d", self.mode, len(cam_paths))
            return cam_paths
        # Training or validation
        # Include only cameras with paired RGB images
        cam_paths = []
        for cam_path in sorted(gfile.Glob(join(trainvali_cam_dir, '*.pb'))):
            id_ = self._parse_id(cam_path)
            img_path = join(trainvali_img_dir, id_ + '.png')
            if gfile.Exists(img_path):
                cam_paths.append(cam_path)
                self.campath2imgpath[cam_path] = img_path
            else:
                logger.warning((
                    "Skipping camera\n\t%s\nbecause its paried RGB image"
                    "\n\t%s\ndoesn't exist"), cam_path, img_path)
        # Training-validation split
        cam_paths_split = []
        for cam_path in cam_paths:
            id_ = self._parse_id(cam_path)
            if (self.mode == 'vali' and id_ in holdout) or \
                    (self.mode != 'vali' and id_ not in holdout):
                cam_paths_split.append(cam_path)
        logger.info("Number of %s views: %d", self.mode, len(cam_paths_split))
        return cam_paths_split

    @staticmethod
    def _parse_id(cam_path):
        return basename(cam_path)[:-3] # strips '.pb'

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, rayo, rayd, rgb, alpha):
        """Records image dimensions and samples rays.
        """
        hw = tf.shape(rgb)[:2]
        rayo, rayd, rgb, alpha = self._sample_rays(rayo, rayd, rgb, alpha)
        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        return id_, hw, rayo, rayd, rgb, alpha

    def _sample_rays(self, rayo, rayd, rgb, alpha):
        # Shortcircuit if need all rays
        if self.mode in ('vali', 'test') or self.always_all_rays:
            rayo = tf.reshape(rayo, (-1, 3))
            rayd = tf.reshape(rayd, (-1, 3))
            rgb = tf.reshape(rgb, (-1, 3))
            alpha = tf.reshape(alpha, (-1,))
            return rayo, rayd, rgb, alpha
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
        alpha = tf.gather_nd(alpha, select_ind)
        return rayo, rayd, rgb, alpha

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, rayo, rayd, rgb, alpha = tf.py_function(
            self._load_data, [path],
            (tf.string, tf.float32, tf.float32, tf.float32, tf.float32))
        return id_, rayo, rayd, rgb, alpha

    def _load_data(self, cam_path):
        imh = self.config.getint('DEFAULT', 'imh')
        # Load camera
        if not isinstance(cam_path, str):
            cam_path = cam_path.numpy().decode()
        id_ = self._parse_id(cam_path)
        cam = ioutil.load_sfm_cam(cam_path)
        # Shortcircuit if testing
        if self.mode == 'test':
            # No RGB for figuring out the image size
            cam_h = cam.ImageShape()[0]
            imw = int(imh / cam_h * cam.ImageShape()[1])
            rgb = tf.zeros((imh, imw, 3)) # placeholder
            alpha = tf.ones((imh, imw)) # placeholder
            if imh != cam_h:
                cam = camutil.resize_cam(cam, imh, imw)
            rayo, rayd = self._gen_rays(cam)
            rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)
            return id_, rayo, rayd, rgb, alpha
        # Training or validation, where each camera has a paired image
        img_path = self.campath2imgpath[cam_path]
        rgba = xm.io.img.load(img_path)
        rgba = xm.img.normalize_uint(rgba)
        if rgba.ndim == 3 and rgba.shape[2] == 3:
            rgba = np.dstack((rgba, np.ones_like(rgba[:, :, 0]))) # NOTE: no
            # alpha, so every pixel is foreground
        # Resize RGB and adjust camera
        if imh != rgba.shape[0]:
            rgba = xm.img.resize(rgba, new_h=imh)
            cam = camutil.resize_cam(cam, *rgba.shape[:2])
        #
        rayo, rayd = self._gen_rays(cam)
        rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32)
        return id_, rayo, rayd, rgb, alpha

    def _gen_rays(self, cam):
        near = self.config.getfloat('DEFAULT', 'near')
        ndc = self.config.getboolean('DEFAULT', 'ndc')
        rayo = np.tile(
            cam.GetPosition()[None, None, :], cam.ImageShape() + (1,))
        rayd = cam.PixelsToRays(cam.GetPixelCenters())
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
            h, w = cam.ImageSizeY(), cam.ImageSizeX()
            f = cam.FocalLength()
            o1 = -1. / (w / (2. * f)) * rayo[..., 0] / rayo[..., 2]
            o2 = -1. / (h / (2. * f)) * rayo[..., 1] / rayo[..., 2]
            o3 = 1. + 2. * near / rayo[..., 2]
            d1 = -1. / (w / (2. * f)) * (
                rayd[..., 0] / rayd[..., 2] - rayo[..., 0] / rayo[..., 2])
            d2 = -1. / (h / (2. * f)) * (
                rayd[..., 1] / rayd[..., 2] - rayo[..., 1] / rayo[..., 2])
            d3 = -2. * near / rayo[..., 2]
            rayo = np.dstack((o1, o2, o3))
            rayd = np.dstack((d1, d2, d3))
        # Normalize ray directions
        rayd = xm.linalg.normalize(rayd, axis=2)
        return rayo, rayd
