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

import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .ns_microfacet_gtlight import Model as BaseModel
from .brdf import Model as BRDFModel
from ..networks import mlp
from ..networks.embedder import Embedder
from ..util import logging as logutil, io as ioutil, geom as geomutil, \
    config as configutil, math as mathutil


logger = logutil.Logger(loggee="models/ns_gtlight")


class Model(BaseModel):
    def __init__(self, config, debug=False):
        # Read BRDF MLP config. as _init_embedder() needs it
        brdf_ckpt = config.get('DEFAULT', 'brdf_model_ckpt')
        brdf_config_path = configutil.get_config_ini(brdf_ckpt)
        self.config_brdf = ioutil.read_config(brdf_config_path)
        self.z_dim = self.config_brdf.getint('DEFAULT', 'z_dim')
        super().__init__(config, debug=debug)
        # Load pretrained BRDF MLP
        self.brdf_model = BRDFModel(self.config_brdf)
        ioutil.restore_model(self.brdf_model, brdf_ckpt)
        self.brdf_model.trainable = False

    def _init_net(self):
        net = super()._init_net()
        net.pop('rough_mlp')
        net.pop('rough_out')
        # BRDF z MLP
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net['brdf_z_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['brdf_z_out'] = mlp.Network([self.z_dim], act=None)
        return net

    def _init_embedder(self):
        embedder = super()._init_embedder()
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        # We need to use the level number used in training the BRDF MLP
        n_freqs_rusink = self.config_brdf.getint('DEFAULT', 'n_freqs')
        # Shortcircuit if not using embedders
        if not pos_enc:
            embedder['rusink'] = tf.identity
            return embedder
        # Rusink. embedder
        kwargs = {
            'incl_input': True,
            'in_dims': 3,
            'log2_max_freq': n_freqs_rusink - 1,
            'n_freqs': n_freqs_rusink,
            'log_sampling': True,
            'periodic_func': [tf.math.sin, tf.math.cos]}
        embedder_rusink = Embedder(**kwargs)
        embedder['rusink'] = embedder_rusink
        return embedder

    def _pred_brdf_at(self, pts):
        normalize_z = self.config_brdf.getboolean('DEFAULT', 'normalize_z')
        mlp_layers = self.net['brdf_z_mlp']
        out_layer = self.net['brdf_z_out']
        brdf_z = tf.zeros((tf.shape(pts)[0], self.z_dim), dtype=tf.float32)
        for i in tf.range(0, tf.shape(pts)[0], self.mlp_chunk):
            end_i = tf.math.minimum(tf.shape(pts)[0], i + self.mlp_chunk)
            pts_chunk = pts[i:end_i, :]
            pts_embed = self.embedder['xyz'](pts_chunk)
            chunk = out_layer(mlp_layers(pts_embed))
            brdf_z = tf.tensor_scatter_nd_update(
                brdf_z, tf.range(i, end_i)[:, None], chunk)
        if normalize_z:
            brdf_z = mathutil.safe_l2_normalize(brdf_z, axis=1)
        return brdf_z # NxZ

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        brdf_scale = self.config.getfloat('DEFAULT', 'learned_brdf_scale')
        z = brdf_prop
        world2local = geomutil.gen_world2local(normal)
        # Transform directions into local frames
        vdir = tf.einsum('jkl,jl->jk', world2local, pts2c)
        ldir = tf.einsum('jkl,jnl->jnk', world2local, pts2l)
        # Directions to Rusink.
        ldir_flat = tf.reshape(ldir, (-1, 3))
        vdir_rep = tf.tile(vdir[:, None, :], (1, tf.shape(ldir)[1], 1))
        vdir_flat = tf.reshape(vdir_rep, (-1, 3))
        rusink = geomutil.dir2rusink(ldir_flat, vdir_flat) # NLx3
        # Repeat BRDF Z
        z_rep = tf.tile(z[:, None, :], (1, tf.shape(ldir)[1], 1))
        z_flat = tf.reshape(z_rep, (-1, tf.shape(z)[1]))
        # Mask out back-lit directions for speed
        local_normal = tf.convert_to_tensor((0, 0, 1), dtype=tf.float32)
        local_normal = tf.reshape(local_normal, (3, 1))
        cos = ldir_flat @ local_normal
        front_lit = tf.reshape(cos, (-1,)) > 0
        rusink_fl = rusink[front_lit]
        z_fl = z_flat[front_lit]
        # Predict BRDF values given identities and Rusink.
        mlp_layers = self.brdf_model.net['brdf_mlp']
        out_layer = self.brdf_model.net['brdf_out']
        brdf_fl = tf.zeros((tf.shape(rusink_fl)[0], 1), dtype=tf.float32)
        for i in tf.range(0, tf.shape(rusink_fl)[0], self.mlp_chunk):
            end_i = tf.math.minimum(tf.shape(rusink_fl)[0], i + self.mlp_chunk)
            z_chunk = z_fl[i:end_i, :]
            rusink_chunk = rusink_fl[i:end_i, :]
            rusink_embed = self.embedder['rusink'](rusink_chunk)
            z_rusink = tf.concat((z_chunk, rusink_embed), axis=1)
            chunk = out_layer(mlp_layers(z_rusink))
            brdf_fl = tf.tensor_scatter_nd_update(
                brdf_fl, tf.range(i, end_i)[:, None], chunk)
        # Put front-lit BRDF values back into an all-zero flat tensor, ...
        brdf_flat = tf.scatter_nd(
            tf.where(front_lit), brdf_fl, (tf.shape(front_lit)[0], 1))
        # and then reshape the resultant flat tensor
        spec = tf.reshape(brdf_flat, (tf.shape(ldir)[0], tf.shape(ldir)[1], 1))
        spec = tf.tile(spec, (1, 1, 3)) # becasue they are achromatic
        # Combine specular and Lambertian components
        brdf = albedo[:, None, :] / np.pi + spec * brdf_scale
        return brdf # NxLx3

    def _brdf_prop_as_img(self, brdf_prop):
        """Z in learned BRDF.

        Input and output are both NumPy arrays, not tensors.
        """
        # Get min. and max. from seen BRDF Zs
        seen_z = self.brdf_model.latent_code.z
        seen_z = seen_z.numpy()
        seen_z_rgb = seen_z[:, :3]
        min_ = seen_z_rgb.min()
        max_ = seen_z_rgb.max()
        range_ = max_ - min_
        assert range_ > 0, "Range of seen BRDF Zs is 0"
        # Clip predicted values and scale them to [0, 1]
        z_rgb = brdf_prop[:, :, :3]
        z_rgb = np.clip(z_rgb, min_, max_)
        z_rgb = (z_rgb - min_) / range_
        return z_rgb
