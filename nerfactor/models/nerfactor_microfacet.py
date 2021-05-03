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

# pylint: disable=arguments-differ,bad-super-call

from os.path import basename
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from brdf.microfacet.microfacet import Microfacet
from nerfactor.models.nerfactor import Model as NeRFactorModel
from nerfactor.networks import mlp
from nerfactor.util import logging as logutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil


logger = logutil.Logger(loggee="models/nerfactor_microfacet")


class Model(NeRFactorModel):
    def __init__(self, config, debug=False):
        # BRDF
        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        self.z_dim = 1 # scalar roughness in microfacet
        self.normalize_brdf_z = False
        # Shape
        self.shape_mode = config.get('DEFAULT', 'shape_mode')
        self.shape_model_ckpt = config.get('DEFAULT', 'shape_model_ckpt')
        shape_config_path = configutil.get_config_ini(self.shape_model_ckpt)
        if self.shape_mode in ('nerf', 'scratch'):
            self.config_shape = None
        else:
            self.config_shape = ioutil.read_config(shape_config_path)
        # By now we have all attributes required by grandparent init.
        super(NeRFactorModel, self).__init__(config, debug=debug)
        # BRDF
        self.albedo_smooth_weight = config.getfloat(
            'DEFAULT', 'albedo_smooth_weight')
        self.brdf_smooth_weight = config.getfloat(
            'DEFAULT', 'brdf_smooth_weight')
        # Lighting
        self._light = None # see the light property
        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        lxyz, lareas = gen_light_xyz(*self.light_res)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        self.lareas = tf.convert_to_tensor(lareas, dtype=tf.float32)
        # Novel lighting conditions for relighting at test time:
        olat_inten = self.config.getfloat('DEFAULT', 'olat_inten', fallback=200)
        ambi_inten = self.config.getfloat(
            'DEFAULT', 'ambient_inten', fallback=0)
        # (1) OLAT
        novel_olat = OrderedDict()
        light_shape = self.light_res + (3,)
        if self.white_bg:
            # Add some ambient lighting to better match perception
            ambient = ambi_inten * tf.ones(light_shape, dtype=tf.float32)
        else:
            ambient = tf.zeros(light_shape, dtype=tf.float32)
        for i in range(2 if self.debug else self.light_res[0]):
            for j in range(2 if self.debug else self.light_res[1]):
                one_hot = tutil.one_hot_img(*ambient.shape, i, j)
                envmap = olat_inten * one_hot + ambient
                novel_olat['%04d-%04d' % (i, j)] = envmap
        self.novel_olat = novel_olat
        # (2) Light probes
        novel_probes = OrderedDict()
        test_envmap_dir = self.config.get('DEFAULT', 'test_envmap_dir')
        for path in xm.os.sortglob(test_envmap_dir, '*.hdr'):
            name = basename(path)[:-len('.hdr')]
            envmap = self._load_light(path)
            novel_probes[name] = envmap
        self.novel_probes = novel_probes
        # Tonemap and visualize these novel lighting conditions
        self.embed_light_h = self.config.getint(
            'DEFAULT', 'embed_light_h', fallback=32)
        self.novel_olat_uint = {}
        for k, v in self.novel_olat.items():
            vis_light = lightutil.vis_light(v, h=self.embed_light_h)
            self.novel_olat_uint[k] = vis_light
        self.novel_probes_uint = {}
        for k, v in self.novel_probes.items():
            vis_light = lightutil.vis_light(v, h=self.embed_light_h)
            self.novel_probes_uint[k] = vis_light
        # PSNR calculator
        self.psnr = xm.metric.PSNR('uint8')

    def _init_embedder(self):
        # Use grandparent's embedders, not parent's, since we don't need
        # the embedder for BRDF coordinates
        embedder = super(NeRFactorModel, self)._init_embedder()
        return embedder

    def _init_net(self):
        net = super()._init_net()
        # Override the roughness MLP output layer to add sigmoid so that [0, 1]
        if self.pred_brdf:
            net['brdf_z_out'] = mlp.Network(
                [self.z_dim], act=['sigmoid']) # [0, 1]
        return net

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        """Fixed to microfacet (GGX).
        """
        rough = brdf_prop
        fresnel_f0 = self.config.getfloat('DEFAULT', 'fresnel_f0')
        microfacet = Microfacet(f0=fresnel_f0)
        brdf = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=rough)
        brdf = tf.debugging.check_numerics(brdf, "BRDF")
        return brdf # NxLx3

    def _brdf_prop_as_img(self, brdf_prop):
        """Roughness in the microfacet BRDF.

        Input and output are both NumPy arrays, not tensors.
        """
        z_rgb = np.concatenate([brdf_prop] * 3, axis=2)
        return z_rgb
