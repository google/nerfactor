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

# pylint: disable=relative-beyond-top-level,arguments-differ

from os.path import join
import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.renderer import gen_light_xyz

from .base import Model as BaseModel
from .light import Model as LightModel
from ..networks.layers import LatentCode
from ..util import logging as logutil, io as ioutil, config as configutil, \
    light as lightutil, img as imgutil


logger = logutil.Logger(loggee="models/light_findz")


class Model(BaseModel):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)
        # Load pretrained light MLP
        light_ckpt = config.get('DEFAULT', 'light_model_ckpt')
        light_config_path = configutil.get_config_ini(light_ckpt)
        config_light = ioutil.read_config(light_config_path)
        self.light_model = LightModel(config_light)
        ioutil.restore_model(self.light_model, light_ckpt)
        self.light_model.trainable = False
        # Init. latent code to optimize
        z_dim = config_light.getint('DEFAULT', 'z_dim')
        z_gauss_mean = config_light.getfloat('DEFAULT', 'z_gauss_mean')
        z_gauss_std = config_light.getfloat('DEFAULT', 'z_gauss_std')
        # z_gauss_std = 0.5 # standard deviation derived from learned codes
        normalize_z = config_light.getboolean('DEFAULT', 'normalize_z')
        self.light_code = LatentCode( # to be optimized
            1, z_dim, mean=z_gauss_mean, std=z_gauss_std, normalize=normalize_z)
        # self.light_code.z = tf.convert_to_tensor( # GT for 3072
        #     [[-0.01786612, -0.92128205, -1.7327819 ]], dtype=tf.float32)
        # self.light_code.z = tf.convert_to_tensor( # mean of learned codes
        #     [[-0.00421232,  0.0112169 ,  0.05657239]], dtype=tf.float32)
        # Lat.-long. locations to query at
        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        xyz, _ = gen_light_xyz(*self.light_res)
        xyz_flat = np.reshape(xyz, (-1, 3))
        rlatlng = xm.geometry.sph.cart2sph(xyz_flat)
        latlng = rlatlng[:, 1:]
        self.latlng = tf.convert_to_tensor(latlng, dtype=tf.float32) # 2D
        # Load the probe we are fitting to
        target_light_path = self.config.get('DEFAULT', 'target_light')
        target_light = xm.io.hdr.read(target_light_path)
        target_light = tf.convert_to_tensor(target_light, dtype=tf.float32)
        self.target_light = imgutil.resize(target_light, new_h=light_h)

    def call(self, batch, mode='train'):
        del batch
        # Predict light code
        z = self.light_code(0) # there is only one global z
        z = tf.tile(z, (tf.shape(self.latlng)[0], 1))
        pred_light = self.light_model.eval_light_at(z, self.latlng) # 2D
        # For loss computation
        pred = {'light': pred_light}
        gt = {'light': tf.reshape(self.target_light, tf.shape(pred_light))}
        loss_kwargs = {}
        # To visualize
        to_vis = {'z': z}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def compute_loss(self, pred, gt, **kwargs):
        loss_transform = self.config.get('DEFAULT', 'loss_transform')
        if loss_transform.lower() == 'none':
            f = tf.identity
        elif loss_transform == 'log':
            f = tf.math.log
        elif loss_transform == 'divide':
            f = lambda x: x / (x + 1.) # noqa
        else:
            raise NotImplementedError(loss_transform)
        loss = tf.keras.losses.MSE(f(gt['light']), f(pred['light']))
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            light_vis_h=256):
        current_z = data_dict['z']
        current_z = current_z.numpy()
        current_z = current_z[0, :]
        print("Current z: %s" % current_z)
        # Visualize probes
        pred_light = data_dict['pred_light']
        gt_light = self.target_light
        pred_light = pred_light.numpy()
        gt_light = gt_light.numpy()
        pred_light = np.reshape(pred_light, self.light_res + (3,))
        gt_light = np.reshape(gt_light, self.light_res + (3,))
        pred_path = join(outdir, 'pred.png')
        gt_path = join(outdir, 'gt.png')
        lightutil.vis_light(pred_light, outpath=pred_path, h=light_vis_h)
        lightutil.vis_light(gt_light, outpath=gt_path, h=light_vis_h)

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train', fps=12):
        pass
