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

from os.path import join, dirname
import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.renderer import gen_light_xyz

from .ns_microfacet_gtlight import Model as BaseModel
from .light import Model as LightModel
from ..networks.layers import LatentCode
from ..util import logging as logutil, io as ioutil, config as configutil, \
    light as lightutil


logger = logutil.Logger(loggee="models/ns_microfacet")


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
        normalize_z = config_light.getboolean('DEFAULT', 'normalize_z')
        self.light_code = LatentCode( # to be optimized
            1, z_dim, mean=z_gauss_mean, std=z_gauss_std, normalize=normalize_z)
        # Lat.-long. locations to query at
        xyz, _ = gen_light_xyz(*self.light_res)
        xyz_flat = np.reshape(xyz, (-1, 3))
        rlatlng = xm.geometry.sph.cart2sph(xyz_flat)
        latlng = rlatlng[:, 1:]
        self.latlng = tf.convert_to_tensor(latlng, dtype=tf.float32) # 2D

    @property
    def light(self):
        light_scale = self.config.getfloat('DEFAULT', 'light_scale')
        # Predict light code
        z = self.light_code(0) # there is only one global z
        z = tf.tile(z, (tf.shape(self.latlng)[0], 1))
        radi = self.light_model.eval_light_at(z, self.latlng) # 2D
        light = tf.reshape(radi, self.light_res + (3,)) # 3D
        light = light_scale * light
        return light # 3D

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            light_vis_h=256):
        # Visualize estimated lighting
        if mode == 'vali':
            # The same for all batches/views, so do it just once
            light_vis_path = join(dirname(outdir), 'pred_light.png')
            if not gfile.Exists(light_vis_path):
                lightutil.vis_light(
                    self.light, outpath=light_vis_path, h=light_vis_h)
        # Do what parent does
        super().vis_batch(data_dict, outdir, mode=mode, dump_raw_to=dump_raw_to)
