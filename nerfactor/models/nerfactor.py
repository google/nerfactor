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

# pylint: disable=arguments-differ

from os.path import basename, dirname, join, exists
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.models.shape import Model as ShapeModel
from nerfactor.models.brdf import Model as BRDFModel
from nerfactor.networks import mlp
from nerfactor.networks.embedder import Embedder
from nerfactor.util import logging as logutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil, img as imgutil, \
    math as mathutil, geom as geomutil


logger = logutil.Logger(loggee="models/nerfactor")


class Model(ShapeModel):
    def __init__(self, config, debug=False):
        # BRDF
        brdf_ckpt = config.get('DEFAULT', 'brdf_model_ckpt')
        brdf_config_path = configutil.get_config_ini(brdf_ckpt)
        self.config_brdf = ioutil.read_config(brdf_config_path)
        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        self.z_dim = self.config_brdf.getint('DEFAULT', 'z_dim')
        self.normalize_brdf_z = self.config_brdf.getboolean(
            'DEFAULT', 'normalize_z')
        # Shape
        self.shape_mode = config.get('DEFAULT', 'shape_mode')
        self.shape_model_ckpt = config.get('DEFAULT', 'shape_model_ckpt')
        shape_config_path = configutil.get_config_ini(self.shape_model_ckpt)
        if self.shape_mode in ('nerf', 'scratch'):
            self.config_shape = None
        else:
            self.config_shape = ioutil.read_config(shape_config_path)
        # By now we have all attributes required by parent init.
        super().__init__(config, debug=debug)
        # BRDF
        self.albedo_smooth_weight = config.getfloat(
            'DEFAULT', 'albedo_smooth_weight')
        self.brdf_smooth_weight = config.getfloat(
            'DEFAULT', 'brdf_smooth_weight')
        self.brdf_model = BRDFModel(self.config_brdf)
        ioutil.restore_model(self.brdf_model, brdf_ckpt)
        self.brdf_model.trainable = False
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

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net = {}
        # Albedo
        net['albedo_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['albedo_out'] = mlp.Network([3], act=['sigmoid']) # [0, 1]
        # BRDF Z
        if self.pred_brdf:
            net['brdf_z_mlp'] = mlp.Network(
                [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
                skip_at=[mlp_skip_at])
            net['brdf_z_out'] = mlp.Network([self.z_dim], act=None)
        # Training from scratch, finetuning, or just using NeRF geometry?
        if self.shape_mode == 'scratch':
            net['normal_mlp'] = mlp.Network(
                [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
                skip_at=[mlp_skip_at])
            net['normal_out'] = mlp.Network(
                [3], act=None) # normalized elsewhere
            net['lvis_mlp'] = mlp.Network(
                [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
                skip_at=[mlp_skip_at])
            net['lvis_out'] = mlp.Network([1], act=['sigmoid']) # [0, 1]
        elif self.shape_mode in ('frozen', 'finetune'):
            shape_model = ShapeModel(self.config_shape)
            ioutil.restore_model(shape_model, self.shape_model_ckpt)
            shape_model.trainable = self.shape_mode == 'finetune'
            net['normal_mlp'] = shape_model.net['normal_mlp']
            net['normal_out'] = shape_model.net['normal_out']
            net['lvis_mlp'] = shape_model.net['lvis_mlp']
            net['lvis_out'] = shape_model.net['lvis_out']
        elif self.shape_mode == 'nerf':
            pass
        else:
            raise ValueError(self.shape_mode)
        return net

    def _load_light(self, path):
        ext = basename(path).split('.')[-1]
        if ext == 'exr':
            light = ioutil.load_exr(path)
            arr = np.dstack((light['R'], light['G'], light['B']))
        elif ext == 'hdr':
            arr = xm.io.hdr.read(path)
        else:
            raise NotImplementedError(ext)
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        resized = imgutil.resize(tensor, new_h=self.light_res[0])
        return resized

    def call(
            self, batch, mode='train', relight_olat=False, relight_probes=False,
            albedo_scales=None, albedo_override=None, brdf_z_override=None):
        xyz_jitter_std = self.config.getfloat('DEFAULT', 'xyz_jitter_std')
        self._validate_mode(mode)
        id_, hw, rayo, _, rgb, alpha, xyz, normal, lvis = batch
        # Mask out 100% background
        mask = alpha[:, 0] > 0
        rayo = tf.boolean_mask(rayo, mask)
        rgb = tf.boolean_mask(rgb, mask)
        xyz = tf.boolean_mask(xyz, mask)
        normal = tf.boolean_mask(normal, mask)
        lvis = tf.boolean_mask(lvis, mask)
        # Directions
        surf2l = self._calc_ldir(xyz)
        surf2c = self._calc_vdir(rayo, xyz)
        # Jitter XYZs
        if xyz_jitter_std > 0:
            xyz_noise = tf.random.normal(tf.shape(xyz), stddev=xyz_jitter_std)
        else:
            xyz_noise = None
        # ------ Normals
        if self.shape_mode == 'nerf':
            normal_pred = normal
            normal_jitter = None
        else:
            normal_pred = self._pred_normal_at(xyz)
            if xyz_noise is not None and self.normal_smooth_weight > 0:
                normal_jitter = self._pred_normal_at(xyz + xyz_noise)
            else:
                normal_jitter = None
        normal_pred = mathutil.safe_l2_normalize(normal_pred, axis=1)
        if normal_jitter is not None:
            normal_jitter = mathutil.safe_l2_normalize(normal_jitter, axis=1)
        # NOTE: rayd and normal_pred must be normalized
        # ------ Light visibility
        if self.shape_mode == 'nerf':
            lvis_pred = lvis
            lvis_jitter = None
        else:
            lvis_pred = self._pred_lvis_at(xyz, surf2l)
            if xyz_noise is not None and self.lvis_smooth_weight > 0:
                lvis_jitter = self._pred_lvis_at(xyz + xyz_noise, surf2l)
            else:
                lvis_jitter = None
        # ------ Albedo
        albedo = self._pred_albedo_at(xyz)
        if xyz_noise is not None and self.albedo_smooth_weight > 0:
            albedo_jitter = self._pred_albedo_at(xyz + xyz_noise)
        else:
            albedo_jitter = None
        if albedo_scales is not None:
            albedo = tf.reshape(albedo_scales, (1, 3)) * albedo
        if albedo_override is not None:
            albedo_override = tf.reshape(albedo_override, (1, 3))
            albedo = tf.tile(albedo_override, (tf.shape(albedo)[0], 1))
        # ------ BRDFs
        if self.pred_brdf:
            brdf_prop = self._pred_brdf_at(xyz)
            if xyz_noise is not None and self.brdf_smooth_weight > 0:
                brdf_prop_jitter = self._pred_brdf_at(xyz + xyz_noise)
            else:
                brdf_prop_jitter = None
            if self.normalize_brdf_z:
                brdf_prop = mathutil.safe_l2_normalize(brdf_prop, axis=1)
                if brdf_prop_jitter is not None:
                    brdf_prop_jitter = mathutil.safe_l2_normalize(
                        brdf_prop_jitter, axis=1)
        else:
            brdf_prop = self._get_default_brdf_at(xyz)
            brdf_prop_jitter = None
        if brdf_z_override is not None:
            brdf_z_override = tf.reshape(
                brdf_z_override, (1, tf.shape(brdf_prop)[1]))
            brdf_prop = tf.tile(brdf_z_override, (tf.shape(brdf_prop)[0], 1))
        brdf = self._eval_brdf_at(
            surf2l, surf2c, normal_pred, albedo, brdf_prop) # NxLx3
        # ------ Rendering equation
        rgb_pred, rgb_olat, rgb_probes = self._render( # all Nx3
            lvis_pred, brdf, surf2l, normal_pred, relight_olat=relight_olat,
            relight_probes=relight_probes)
        # Put values back into the full shape
        ind = tf.where(mask)
        n = tf.shape(alpha)[0] # total number of rays
        l = tf.shape(lvis_pred)[1] # total number of light directions
        rgb_pred = tf.scatter_nd(ind, rgb_pred, (n, 3))
        normal_pred = tf.scatter_nd(ind, normal_pred, (n, 3))
        lvis_pred = tf.scatter_nd(ind, lvis_pred, (n, l))
        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        brdf_prop = tf.scatter_nd(ind, brdf_prop, (n, self.z_dim))
        if rgb_olat is not None:
            rgb_olat = tf.scatter_nd(
                ind, rgb_olat, (n, len(self.novel_olat), 3))
        if rgb_probes is not None:
            rgb_probes = tf.scatter_nd(
                ind, rgb_probes, (n, len(self.novel_probes), 3))
        rgb = tf.scatter_nd(ind, rgb, (n, 3))
        normal = tf.scatter_nd(ind, normal, (n, 3))
        lvis = tf.scatter_nd(ind, lvis, (n, l))
        normal_jitter = tf.scatter_nd(ind, normal_jitter, (n, 3))
        lvis_jitter = tf.scatter_nd(ind, lvis_jitter, (n, l))
        brdf_prop_jitter = tf.scatter_nd(ind, brdf_prop_jitter, (n, self.z_dim))
        albedo_jitter = tf.scatter_nd(ind, albedo_jitter, (n, 3))
        # ------ Loss
        pred = {
            'rgb': rgb_pred, 'normal': normal_pred, 'lvis': lvis_pred,
            'albedo': albedo, 'brdf': brdf_prop}
        if rgb_olat is not None:
            pred['rgb_olat'] = rgb_olat
        if rgb_probes is not None:
            pred['rgb_probes'] = rgb_probes
        gt = {'rgb': rgb, 'normal': normal, 'lvis': lvis, 'alpha': alpha}
        loss_kwargs = {
            'mode': mode, 'normal_jitter': normal_jitter,
            'lvis_jitter': lvis_jitter, 'brdf_prop_jitter': brdf_prop_jitter,
            'albedo_jitter': albedo_jitter}
        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def _render(
            self, light_vis, brdf, l, n,
            relight_olat=False, relight_probes=False,
            white_light_override=False, white_lvis_override=False):
        linear2srgb = self.config.getboolean('DEFAULT', 'linear2srgb')
        light = self.light
        if white_light_override:
            light = np.ones_like(self.light)
        if white_lvis_override:
            light_vis = np.ones_like(light_vis)
        cos = tf.einsum('ijk,ik->ij', l, n) # NxL
        # Areas for intergration
        areas = tf.reshape(self.lareas, (1, -1, 1)) # 1xLx1
        # NOTE: unnecessary if light_vis already encodes it, but won't hurt
        front_lit = tf.cast(cos > 0, tf.float32)
        lvis = front_lit * light_vis # NxL

        def integrate(light):
            light_flat = tf.reshape(light, (-1, 3)) # Lx3
            light = lvis[:, :, None] * light_flat[None, :, :] # NxLx3
            light_pix_contrib = brdf * light * cos[:, :, None] * areas # NxLx3
            rgb = tf.reduce_sum(light_pix_contrib, axis=1) # Nx3
            # Tonemapping
            rgb = tf.clip_by_value(rgb, 0., 1.) # NOTE
            # Colorspace transform
            if linear2srgb:
                rgb = imgutil.linear2srgb(rgb)
            return rgb

        # ------ Render under original lighting
        rgb = integrate(light)
        # ------ Continue to render OLAT-relit results
        rgb_olat = None
        if relight_olat:
            rgb_olat = []
            for _, light in self.novel_olat.items():
                rgb_relit = integrate(light)
                rgb_olat.append(rgb_relit)
            rgb_olat = tf.concat([x[:, None, :] for x in rgb_olat], axis=1)
            rgb_olat = tf.debugging.check_numerics(rgb_olat, "OLAT Renders")
        # ------ Continue to render light probe-relit results
        rgb_probes = None
        if relight_probes:
            rgb_probes = []
            for _, light in self.novel_probes.items():
                rgb_relit = integrate(light)
                rgb_probes.append(rgb_relit)
            rgb_probes = tf.concat([x[:, None, :] for x in rgb_probes], axis=1)
            rgb_probes = tf.debugging.check_numerics(
                rgb_probes, "Light Probe Renders")
        return rgb, rgb_olat, rgb_probes # Nx3

    @property
    def light(self):
        if self._light is None: # initialize just once
            maxv = self.config.getfloat('DEFAULT', 'light_init_max')
            light = tf.random.uniform(
                self.light_res + (3,), minval=0., maxval=maxv)
            self._light = tf.Variable(light, trainable=True)
        # No negative light
        return tf.clip_by_value(self._light, 0., np.inf) # 3D

    def _pred_albedo_at(self, pts):
        # Given that albedo generally ranges from 0.1 to 0.8
        albedo_scale = self.config.getfloat(
            'DEFAULT', 'albedo_slope', fallback=0.7)
        albedo_bias = self.config.getfloat(
            'DEFAULT', 'albedo_bias', fallback=0.1)
        mlp_layers = self.net['albedo_mlp']
        out_layer = self.net['albedo_out'] # output in [0, 1]
        embedder = self.embedder['xyz']

        def chunk_func(surf):
            surf_embed = embedder(surf)
            albedo = out_layer(mlp_layers(surf_embed))
            return albedo

        albedo = self.chunk_apply(chunk_func, pts, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias # [bias, scale + bias]
        albedo = tf.debugging.check_numerics(albedo, "Albedo")
        return albedo # Nx3

    def _pred_brdf_at(self, pts):
        mlp_layers = self.net['brdf_z_mlp']
        out_layer = self.net['brdf_z_out']
        embedder = self.embedder['xyz']

        def chunk_func(surf):
            surf_embed = embedder(surf)
            brdf_z = out_layer(mlp_layers(surf_embed))
            return brdf_z

        brdf_z = self.chunk_apply(chunk_func, pts, self.z_dim, self.mlp_chunk)
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
        embedder = self.embedder['rusink']

        def chunk_func(rusink_z):
            rusink, z = rusink_z[:, :3], rusink_z[:, 3:]
            rusink_embed = embedder(rusink)
            z_rusink = tf.concat((z, rusink_embed), axis=1)
            # Strange that shape can't be inferred from the restored
            # `self.brdf_model`, so we set it manually
            z_rusink = tf.ensure_shape(
                z_rusink, (None, self.z_dim + embedder.out_dims))
            brdf = out_layer(mlp_layers(z_rusink))
            return brdf

        rusink_z = tf.concat((rusink_fl, z_fl), 1)
        brdf_fl = self.chunk_apply(chunk_func, rusink_z, 1, self.mlp_chunk)
        # Put front-lit BRDF values back into an all-zero flat tensor, ...
        brdf_flat = tf.scatter_nd(
            tf.where(front_lit), brdf_fl, (tf.shape(front_lit)[0], 1))
        # and then reshape the resultant flat tensor
        spec = tf.reshape(brdf_flat, (tf.shape(ldir)[0], tf.shape(ldir)[1], 1))
        spec = tf.tile(spec, (1, 1, 3)) # becasue they are achromatic
        # Combine specular and Lambertian components
        brdf = albedo[:, None, :] / np.pi + spec * brdf_scale
        return brdf # NxLx3

    def compute_loss(self, pred, gt, **kwargs):
        """Additional priors on light probes.
        """
        normal_loss_weight = self.config.getfloat(
            'DEFAULT', 'normal_loss_weight')
        lvis_loss_weight = self.config.getfloat('DEFAULT', 'lvis_loss_weight')
        smooth_use_l1 = self.config.getboolean('DEFAULT', 'smooth_use_l1')
        light_tv_weight = self.config.getfloat('DEFAULT', 'light_tv_weight')
        light_achro_weight = self.config.getfloat(
            'DEFAULT', 'light_achro_weight')
        smooth_loss = tf.keras.losses.MAE if smooth_use_l1 \
            else tf.keras.losses.MSE
        #
        mode = kwargs.pop('mode')
        normal_jitter = kwargs.pop('normal_jitter')
        lvis_jitter = kwargs.pop('lvis_jitter')
        albedo_jitter = kwargs.pop('albedo_jitter')
        brdf_prop_jitter = kwargs.pop('brdf_prop_jitter')
        #
        alpha, rgb_gt = gt['alpha'], gt['rgb']
        rgb_pred = pred['rgb']
        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
        albedo_pred = pred['albedo']
        brdf_prop_pred = pred['brdf']
        # Composite prediction and ground truth onto backgrounds
        bg = tf.ones_like(rgb_gt) if self.white_bg else tf.zeros_like(rgb_gt)
        rgb_pred = imgutil.alpha_blend(rgb_pred, alpha, tensor2=bg)
        rgb_gt = imgutil.alpha_blend(rgb_gt, alpha, tensor2=bg)
        bg = tf.ones_like(normal_gt) if self.white_bg \
            else tf.zeros_like(normal_gt)
        normal_pred = imgutil.alpha_blend(normal_pred, alpha, tensor2=bg)
        normal_gt = imgutil.alpha_blend(normal_gt, alpha, tensor2=bg)
        bg = tf.ones_like(lvis_gt) if self.white_bg else tf.zeros_like(lvis_gt)
        lvis_pred = imgutil.alpha_blend(lvis_pred, alpha, tensor2=bg)
        lvis_gt = imgutil.alpha_blend(lvis_gt, alpha, tensor2=bg)
        # RGB recon. loss is always here
        loss = tf.keras.losses.MSE(rgb_gt, rgb_pred) # N
        # If validation, just MSE -- return immediately
        if mode == 'vali':
            return loss
        # If we modify the geometry
        if self.shape_mode in ('scratch', 'finetune'):
            # Predicted values should be close to NeRF values
            normal_loss = tf.keras.losses.MSE(normal_gt, normal_pred) # N
            lvis_loss = tf.keras.losses.MSE(lvis_gt, lvis_pred) # N
            loss += normal_loss_weight * normal_loss
            loss += lvis_loss_weight * lvis_loss
            # Predicted values should be smooth
            if normal_jitter is not None:
                normal_smooth_loss = smooth_loss(normal_pred, normal_jitter) # N
                loss += self.normal_smooth_weight * normal_smooth_loss
            if lvis_jitter is not None:
                lvis_smooth_loss = smooth_loss(lvis_pred, lvis_jitter) # N
                loss += self.lvis_smooth_weight * lvis_smooth_loss
        # Albedo should be smooth
        if albedo_jitter is not None:
            albedo_smooth_loss = smooth_loss(albedo_pred, albedo_jitter) # N
            loss += self.albedo_smooth_weight * albedo_smooth_loss
        # BRDF property should be smooth
        if brdf_prop_jitter is not None:
            brdf_smooth_loss = smooth_loss(brdf_prop_pred, brdf_prop_jitter) # N
            loss += self.brdf_smooth_weight * brdf_smooth_loss
        # Light should be smooth
        if mode == 'train':
            light = self.light
            # Spatial TV penalty
            if light_tv_weight > 0:
                dx = light - tf.roll(light, 1, 1)
                dy = light - tf.roll(light, 1, 0)
                tv = tf.reduce_sum(dx ** 2 + dy ** 2)
                loss += light_tv_weight * tv
            # Cross-channel TV penalty
            if light_achro_weight > 0:
                dc = light - tf.roll(light, 1, 2)
                tv = tf.reduce_sum(dc ** 2)
                loss += light_achro_weight * tv
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss

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

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            light_vis_h=256, olat_vis=False, alpha_thres=0.8):
        # Visualize estimated lighting
        if mode == 'vali':
            # The same for all batches/views, so do it just once
            light_vis_path = join(dirname(outdir), 'pred_light.png')
            if not exists(light_vis_path):
                lightutil.vis_light(
                    self.light, outpath=light_vis_path, h=light_vis_h)
        # Do what parent does
        self._validate_mode(mode)
        # Shortcircuit if training because rays are randomly sampled and
        # therefore very likely don't form a complete image
        if mode == 'train':
            return
        hw = data_dict.pop('hw')[0, :]
        hw = tuple(hw.numpy())
        id_ = data_dict.pop('id')[0]
        id_ = tutil.eager_tensor_to_str(id_)
        # To NumPy and reshape back to images
        for k, v in data_dict.items():
            if v is None:
                continue # no-op
            v_ = v.numpy()
            if k in ('pred_rgb_olat', 'pred_rgb_probes'):
                v_ = v_.reshape(hw + (v_.shape[1], 3))
            elif k.endswith(('rgb', 'albedo', 'normal')):
                v_ = v_.reshape(hw + (3,))
            elif k.endswith(('occu', 'depth', 'disp', 'alpha')):
                v_ = v_.reshape(hw)
            elif k.endswith('brdf'):
                v_ = v_.reshape(hw + (-1,))
            elif k.endswith('lvis'):
                v_ = v_.reshape(hw + (v_.shape[1],))
            else:
                raise NotImplementedError(k)
            data_dict[k] = v_

        def composite_on_avg_light(render, light_uint):
            # Compute average lighting
            lareas = self.lareas.numpy()
            lareas_upper = lareas[:(lareas.shape[0] // 2), :]
            weights = np.dstack([lareas_upper] * 3)
            light = xm.img.normalize_uint(light_uint) # now float
            light = xm.img.resize(light, new_h=lareas.shape[0])
            light_upper = light[:(light.shape[0] // 2), :, :]
            avg_light = np.average( # (3,)
                light_upper, axis=(0, 1), weights=weights)
            # Composite results on average lighting background
            bg = np.tile(
                avg_light[None, None, :], v_relit.shape[:2] + (1,))
            img = xm.img.alpha_blend(render, alpha, bg)
            return img

        # Write images
        img_dict = {}
        alpha = data_dict['gt_alpha']
        alpha[alpha < alpha_thres] = 0 # stricter compositing
        for k, v in data_dict.items():
            # OLAT-relit RGB
            if k == 'pred_rgb_olat': # HxWxLx3
                if v is None:
                    continue
                olat_names = list(self.novel_olat.keys())
                olat_first_n = np.prod(self.light_res) // 2 # top half only
                olat_n = 0
                for i, lname in enumerate(
                        tqdm(olat_names, desc="Writing OLAT-Relit Results")):
                    # Skip visiualization if enough OLATs
                    if olat_n >= olat_first_n:
                        continue
                    else:
                        olat_n += 1
                    #
                    k_relit = k + '_' + lname
                    v_relit = v[:, :, i, :]
                    light_uint = self.novel_olat_uint[lname]
                    img = composite_on_avg_light(v_relit, light_uint)
                    img_dict[k_relit] = xm.io.img.write_arr(
                        img, join(outdir, k_relit + '.png'), clip=True)
            # Light probe-relit RGB
            elif k == 'pred_rgb_probes': # HxWxLx3
                if v is None:
                    continue
                probe_names = list(self.novel_probes.keys())
                for i, lname in enumerate(probe_names):
                    k_relit = k + '_' + lname
                    v_relit = v[:, :, i, :]
                    light_uint = self.novel_probes_uint[lname]
                    img = composite_on_avg_light(v_relit, light_uint)
                    img_dict[k_relit] = xm.io.img.write_arr(
                        img, join(outdir, k_relit + '.png'), clip=True)
            # RGB
            elif k.endswith('rgb'): # HxWx3
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Normals
            elif k.endswith('normal'):
                v_ = (v + 1) / 2 # [-1, 1] to [0, 1]
                bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                img = imgutil.alpha_blend(v_, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Albedo
            elif k.endswith('albedo'):
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                v_gamma = v ** (1 / 2.2)
                img = imgutil.alpha_blend(v_gamma, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Light visibility
            elif k.endswith('lvis'):
                mean = np.mean(v, axis=2) # NOTE: average across all lights
                bg = np.ones_like(mean) if self.white_bg \
                    else np.zeros_like(mean)
                img = imgutil.alpha_blend(mean, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
                # Optionally, visualize per-light vis.
                if olat_vis:
                    for i in tqdm(
                            range(4 if self.debug else v.shape[2] // 2), # half
                            desc="Writing Per-Light Visibility (%s)" % k):
                        v_olat = v[:, :, i]
                        ij = np.unravel_index(i, self.light_res)
                        k_olat = k + '_olat_%04d-%04d' % ij
                        img = imgutil.alpha_blend(v_olat, alpha, bg)
                        img_dict[k_olat] = xm.io.img.write_arr(
                            img, join(outdir, k_olat + '.png'), clip=True)
            # BRDF property
            elif k.endswith('brdf'):
                v_ = self._brdf_prop_as_img(v)
                bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                img = imgutil.alpha_blend(v_, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Everything else
            else:
                img = v
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
        # Shortcircuit if testing because there will be no ground truth for
        # us to make .apng comparisons
        if mode == 'test':
            # Write metadata that doesn't require ground truth (e.g., view name)
            metadata = {'id': id_}
            ioutil.write_json(metadata, join(outdir, 'metadata.json'))
            return
        # Make .apng
        put_text_kwargs = {
            'label_top_left_xy': (
                int(self.put_text_param['text_loc_ratio'] * hw[1]),
                int(self.put_text_param['text_loc_ratio'] * hw[0])),
            'font_size': int(self.put_text_param['text_size_ratio'] * hw[0]),
            'font_color': (0, 0, 0) if self.white_bg else (1, 1, 1),
            'font_ttf': self.put_text_param['font_path']}
        im1 = xm.vis.text.put_text(
            img_dict['gt_rgb'], "Ground Truth", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['pred_rgb'], "Prediction", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'pred-vs-gt_rgb.apng'))
        if self.shape_mode != 'nerf':
            im1 = xm.vis.text.put_text(
                img_dict['gt_normal'], "NeRF", **put_text_kwargs)
            im2 = xm.vis.text.put_text(
                img_dict['pred_normal'], "Prediction", **put_text_kwargs)
            xm.vis.anim.make_anim(
                (im1, im2), outpath=join(outdir, 'pred-vs-gt_normal.apng'))
            im1 = xm.vis.text.put_text(
                img_dict['gt_lvis'], "NeRF", **put_text_kwargs)
            im2 = xm.vis.text.put_text(
                img_dict['pred_lvis'], "Prediction", **put_text_kwargs)
            xm.vis.anim.make_anim(
                (im1, im2), outpath=join(outdir, 'pred-vs-gt_lvis.apng'))
        # Write metadata (e.g., view name, PSNR, etc.)
        psnr = self.psnr(img_dict['gt_rgb'], img_dict['pred_rgb'])
        metadata = {'id': id_, 'psnr': psnr}
        ioutil.write_json(metadata, join(outdir, 'metadata.json'))

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train', fps=12):
        viewer_prefix = self.config.get('DEFAULT', 'viewer_prefix')
        self._validate_mode(mode)
        # Shortcircuit if training (same reason as above)
        if mode == 'train':
            return None
        # Validation or testing
        if mode == 'vali':
            outpath = outpref + '.html'
            self._compile_into_webpage(batch_vis_dirs, outpath)
        else:
            outpath = outpref + '.mp4'
            self._compile_into_video(batch_vis_dirs, outpath, fps=fps)
        view_at = viewer_prefix + outpath
        return view_at # to be logged into TensorBoard

    def _compile_into_webpage(self, batch_dirs, out_html):
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            metadata = str(metadata)
            row = [
                metadata,
                join(batch_dir, 'pred-vs-gt_rgb.apng'),
                join(batch_dir, 'pred_rgb.png'),
                join(batch_dir, 'pred_albedo.png'),
                join(batch_dir, 'pred_brdf.png')]
            rowcaps = [
                "Metadata", "RGB", "RGB (pred.)", "Albedo (pred.)",
                "BRDF (pred.)"]
            rowtypes = ['text', 'image', 'image', 'image', 'image']
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_normal.png'))
                rowcaps.append("Normal (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_normal.apng'))
                rowcaps.append("Normal")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_normal.png'))
                rowcaps.append("Normal (pred.)")
                rowtypes.append('image')
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_lvis.png'))
                rowcaps.append("Light Visibility (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_lvis.apng'))
                rowcaps.append("Light Visibility")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_lvis.png'))
                rowcaps.append("Light Visibility (pred.)")
                rowtypes.append('image')
            #
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Write HTML
        bg_color = 'white' if self.white_bg else 'black'
        text_color = 'black' if self.white_bg else 'white'
        html = xm.vis.html.HTML(bgcolor=bg_color, text_color=text_color)
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)

    def _compile_into_video(self, batch_dirs, out_mp4, fps=12):
        batch_dirs = sorted(
            batch_dirs) # assuming batch directory order is the right view order
        if self.debug:
            batch_dirs = batch_dirs[:10]
        # Tonemap and visualize all lighting conditions used
        orig_light_uint = lightutil.vis_light(self.light, h=self.embed_light_h)

        def make_frame(path_dict, light=None, task='viewsyn'):
            # Load predictions
            nn = xm.io.img.load(path_dict['nn'])
            albedo = xm.io.img.load(path_dict['albedo'])
            lvis = xm.io.img.load(path_dict['lvis'])
            normal = xm.io.img.load(path_dict['normal'])
            rgb = xm.io.img.load(path_dict['rgb'])
            brdf = xm.io.img.load(path_dict['brdf'])
            hw = rgb.shape[:2]
            # Optionally, embed the light used to right bottom corner of render
            if light is not None:
                imgutil.frame_image(light, rgb=(1, 1, 1), width=1)
                light_vis_h = int(32 / 256 * hw[0]) # scale light probe size
                light = xm.img.resize(light, new_h=light_vis_h)
                rgb[:light.shape[0], -light.shape[1]:] = light
            # Put labels
            put_text_kwargs = {
                'label_top_left_xy': (
                    int(self.put_text_param['text_loc_ratio'] * hw[1]),
                    int(self.put_text_param['text_loc_ratio'] * hw[0])),
                'font_size': int(
                    self.put_text_param['text_size_ratio'] * hw[0]),
                'font_color': (0, 0, 0) if self.white_bg else (1, 1, 1),
                'font_ttf': self.put_text_param['font_path']}
            nn = xm.vis.text.put_text(nn, "Nearest Input", **put_text_kwargs)
            albedo = xm.vis.text.put_text(albedo, "Albedo", **put_text_kwargs)
            lvis_label = "Light Visibility"
            if task in ('viewsyn', 'simul'):
                lvis_label += " (mean)"
            lvis = xm.vis.text.put_text(lvis, lvis_label, **put_text_kwargs)
            normal = xm.vis.text.put_text(normal, "Normals", **put_text_kwargs)
            rgb = xm.vis.text.put_text(rgb, "Rendering", **put_text_kwargs)
            brdf = xm.vis.text.put_text(brdf, "BRDF", **put_text_kwargs)
            # Make collage
            frame_top = imgutil.hconcat((normal, lvis, nn))
            frame_bottom = imgutil.hconcat((brdf, albedo, rgb))
            frame = imgutil.vconcat((frame_top, frame_bottom))
            return frame

        def get_nearest_input(batch_dir):
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            id_ = metadata['id']
            data_root = self.config.get('DEFAULT', 'data_root')
            nearest_input_path = join(data_root, id_, 'nn.png')
            return nearest_input_path

        # ------ View synthesis
        frames = []
        for batch_dir in tqdm(batch_dirs, desc="View Synthesis"):
            paths = {
                'nn': get_nearest_input(batch_dir),
                'albedo': join(batch_dir, 'pred_albedo.png'),
                'lvis': join(batch_dir, 'pred_lvis.png'), # mean
                'normal': join(batch_dir, 'pred_normal.png'),
                'rgb': join(batch_dir, 'pred_rgb.png'),
                'alpha': join(batch_dir, 'gt_alpha.png'),
                'brdf': join(batch_dir, 'pred_brdf.png')}
            if not ioutil.all_exist(paths):
                logger.warn(
                    "Skipping because of missing files:\n\t%s", batch_dir)
                continue
            frame = make_frame(paths, light=orig_light_uint, task='viewsyn')
            frames.append(frame)
        # ------ Relighting
        relight_view_dir = batch_dirs[-1] # fixed to the final view
        lvis_paths = xm.os.sortglob(relight_view_dir, 'pred_lvis_olat*.png')
        for lvis_path in tqdm(lvis_paths, desc="Final View, OLAT"):
            olat_id = basename(lvis_path)[len('pred_lvis_olat_'):-len('.png')]
            if self.debug and (olat_id not in self.novel_probes_uint):
                continue
            rgb_path = join(relight_view_dir, 'pred_rgb_olat_%s.png' % olat_id)
            paths = {
                'nn': get_nearest_input(relight_view_dir),
                'albedo': join(relight_view_dir, 'pred_albedo.png'),
                'lvis': lvis_path, # per-light
                'normal': join(relight_view_dir, 'pred_normal.png'),
                'rgb': rgb_path,
                'alpha': join(relight_view_dir, 'gt_alpha.png'),
                'brdf': join(relight_view_dir, 'pred_brdf.png')}
            if not ioutil.all_exist(paths):
                logger.warn("Skipping because of missing files: %s", olat_id)
                continue
            light_used = self.novel_olat_uint[olat_id]
            frame = make_frame(paths, light=light_used, task='relight')
            frames.append(frame)
        # ------ Simultaneous
        envmap_names = list(self.novel_probes.keys())
        n_envmaps = len(envmap_names)
        batch_dirs_roundtrip = list(reversed(batch_dirs)) + batch_dirs
        batch_dirs_roundtrip += batch_dirs_roundtrip # 2nd roundtrip
        n_views_per_envmap = len(batch_dirs_roundtrip) / n_envmaps # float
        map_i = 0
        for view_i, batch_dir in enumerate(
                tqdm(batch_dirs_roundtrip, desc="View Roundtrip, IBL")):
            envmap_name = envmap_names[map_i]
            rgb_path = join(batch_dir, 'pred_rgb_probes_%s.png' % envmap_name)
            paths = {
                'nn': get_nearest_input(batch_dir),
                'albedo': join(batch_dir, 'pred_albedo.png'),
                'lvis': join(batch_dir, 'pred_lvis.png'), # mean
                'normal': join(batch_dir, 'pred_normal.png'),
                'rgb': rgb_path,
                'alpha': join(batch_dir, 'gt_alpha.png'),
                'brdf': join(batch_dir, 'pred_brdf.png')}
            if not ioutil.all_exist(paths):
                logger.warn(
                    "Skipping because of missing files:\n\t%s", batch_dir)
                continue
            light_used = self.novel_probes_uint[envmap_name]
            frame = make_frame(paths, light=light_used, task='simul')
            frames.append(frame)
            # Time to switch to the next map?
            if (view_i + 1) > n_views_per_envmap * (map_i + 1):
                map_i += 1
        #
        xm.vis.video.make_video(frames, outpath=out_mp4, fps=fps)
