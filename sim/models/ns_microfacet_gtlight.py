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

from os.path import join, basename
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.renderer import gen_light_xyz
from google3.experimental.users.xiuming.sim.brdf.microfacet.microfacet \
    import Microfacet

from .ns_shape import Model as ShapeModel
from ..networks import mlp
from ..util import logging as logutil, io as ioutil, img as imgutil, \
    config as configutil, tensor as tutil, math as mathutil, light as lightutil


logger = logutil.Logger(loggee="models/ns_microfacet_gtlight")


class Model(ShapeModel):
    def __init__(self, config, debug=False):
        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        # ------ Shape
        self.shape_mode = config.get('DEFAULT', 'shape_mode')
        self.shape_model_ckpt = config.get('DEFAULT', 'shape_model_ckpt')
        shape_config_path = configutil.get_config_ini(self.shape_model_ckpt)
        if self.shape_mode in ('nerf', 'scratch'):
            self.config_shape = None
        else:
            self.config_shape = ioutil.read_config(shape_config_path)
        super().__init__(config, debug=debug)
        # ------ BRDF
        self.albedo_smooth_weight = self.config.getfloat(
            'DEFAULT', 'albedo_smooth_weight')
        self.brdf_smooth_weight = self.config.getfloat(
            'DEFAULT', 'brdf_smooth_weight')
        # ------ Lighting
        self._light = None # see the light property
        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        lxyz, lareas = gen_light_xyz(*self.light_res)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        self.lareas = tf.convert_to_tensor(lareas, dtype=tf.float32)
        # ------ Novel lighting conditions for relighting at test time
        olat_inten = self.config.getfloat('DEFAULT', 'olat_inten', fallback=200)
        ambi_inten = self.config.getfloat(
            'DEFAULT', 'ambient_inten', fallback=0)
        lights_novel = OrderedDict()
        # OLAT
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
                lights_novel['olat-%04d-%04d' % (i, j)] = envmap
        # Environment maps
        test_envmap_dir = self.config.get('DEFAULT', 'test_envmap_dir')
        for exr_path in xm.os.sortglob(test_envmap_dir, '*.exr'):
            name = basename(exr_path)[:-len('.exr')]
            envmap = self._load_light(exr_path)
            lights_novel[name] = envmap
        self.lights_novel = lights_novel
        # Tonemap and visualize these novel lighting conditions
        self.embed_light_h = self.config.getint(
            'DEFAULT', 'embed_light_h', fallback=32)
        self.lights_novel_uint = {}
        for k, v in self.lights_novel.items():
            vis_light = lightutil.vis_light(v, h=self.embed_light_h)
            self.lights_novel_uint[k] = vis_light
        # ------ PSNR calculator
        self.psnr = xm.metric.PSNR('uint8')

    @property
    def light(self):
        if self._light is None: # load just once
            light_path = self.config.get('DEFAULT', 'gt_light')
            light_scale = self.config.getfloat('DEFAULT', 'light_scale')
            gt_light = light_scale * self._load_light(light_path)
            self._light = tf.Variable( # so that we can access it in eager
                gt_light, trainable=False) # (i.e., outside of tf.function)
        return self._light # 3D

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
        # Microfacet roughness?
        if self.pred_brdf:
            net['rough_mlp'] = mlp.Network(
                [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
                skip_at=[mlp_skip_at])
            net['rough_out'] = mlp.Network([1], act=['sigmoid']) # [0, 1]
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

    def call(
            self, batch, mode='train', albedo_scales=None, albedo_override=None,
            brdf_z_override=None):
        xyz_jitter_std = self.config.getfloat('DEFAULT', 'xyz_jitter_std')
        self._validate_mode(mode)
        id_, hw, rayo, _, rgb, alpha, xyz, normal, lvis = batch
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
        relight = mode == 'test'
        rgb_pred = self._render( # Nx3 or Nx(1+L+M)x3
            lvis_pred, brdf, surf2l, normal_pred, relight=relight)
        # ------ Loss
        pred = {
            'rgb': rgb_pred, 'normal': normal_pred, 'lvis': lvis_pred,
            'albedo': albedo, 'brdf': brdf_prop}
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
            self, light_vis, brdf, l, n, relight=False,
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
        if not relight:
            return rgb # Nx3
        # ------ Continue to render extra relit results
        rgb = [rgb] # listify
        for _, light in tqdm(
                self.lights_novel.items(), desc="Rendering relit results"):
            rgb_relit = integrate(light)
            rgb.append(rgb_relit)
        rgb = tf.concat([x[:, None, :] for x in rgb], axis=1)
        rgb = tf.debugging.check_numerics(rgb, "Renders")
        return rgb # Nx(1+L+M)x3

    def _pred_brdf_at(self, pts):
        rough_min = self.config.getfloat('DEFAULT', 'rough_min')
        mlp_layers = self.net['rough_mlp']
        out_layer = self.net['rough_out']
        # Chunk by chunk to avoid OOM
        rough = tf.zeros((tf.shape(pts)[0], 1), dtype=tf.float32)
        for i in tf.range(0, tf.shape(pts)[0], self.mlp_chunk):
            end_i = tf.math.minimum(tf.shape(pts)[0], i + self.mlp_chunk)
            pts_chunk = pts[i:end_i]
            # Positional encoding
            pts_embed = self.embedder['xyz'](pts_chunk)
            # Evaluate MLP
            chunk = out_layer(mlp_layers(pts_embed))
            #
            rough = tf.tensor_scatter_nd_update(
                rough, tf.range(i, end_i)[:, None], chunk)
        # Rescale predicted values to satisfy minimum requirement
        rough = rough * (1. - rough_min) + rough_min # [0, 1] to [d, 1]
        rough = tf.debugging.check_numerics(rough, "Roughness")
        return rough # Nx1

    def _get_default_brdf_at(self, pts):
        default_rough = self.config.getfloat('DEFAULT', 'default_rough')
        rough = default_rough * tf.ones(
            (tf.shape(pts)[0], 1), dtype=pts.dtype)
        return rough

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        """Fixed to microfacet (GGX).
        """
        rough = brdf_prop
        fresnel_f0 = self.config.getfloat('DEFAULT', 'fresnel_f0')
        microfacet = Microfacet(f0=fresnel_f0)
        brdf = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=rough)
        brdf = tf.debugging.check_numerics(brdf, "BRDF")
        return brdf # NxLx3

    def _pred_albedo_at(self, pts):
        # Given that albedo generally ranges from 0.1 to 0.8
        albedo_scale = self.config.getfloat(
            'DEFAULT', 'albedo_slope', fallback=0.7)
        albedo_bias = self.config.getfloat(
            'DEFAULT', 'albedo_bias', fallback=0.1)
        mlp_layers = self.net['albedo_mlp']
        out_layer = self.net['albedo_out'] # output in [0, 1]
        # Chunk by chunk to avoid OOM
        albedo = tf.zeros((tf.shape(pts)[0], 3), dtype=tf.float32)
        for i in tf.range(0, tf.shape(pts)[0], self.mlp_chunk):
            end_i = tf.math.minimum(tf.shape(pts)[0], i + self.mlp_chunk)
            pts_chunk = pts[i:end_i]
            # Positional encoding
            pts_embed = self.embedder['xyz'](pts_chunk)
            # Evaluate MLP
            chunk = out_layer(mlp_layers(pts_embed))
            #
            albedo = tf.tensor_scatter_nd_update(
                albedo, tf.range(i, end_i)[:, None], chunk)
        albedo = albedo_scale * albedo + albedo_bias # [bias, scale + bias]
        albedo = tf.debugging.check_numerics(albedo, "Albedo")
        return albedo # Nx3

    def compute_loss(self, pred, gt, **kwargs):
        """Composites signals onto backgrounds and computes the loss.
        """
        normal_loss_weight = self.config.getfloat(
            'DEFAULT', 'normal_loss_weight')
        lvis_loss_weight = self.config.getfloat('DEFAULT', 'lvis_loss_weight')
        smooth_use_l1 = self.config.getboolean('DEFAULT', 'smooth_use_l1')
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
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss

    def vis_batch(self, data_dict, outdir, mode='train', dump_raw_to=None):
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
            v_ = v.numpy()
            if k.endswith('rgb'):
                if v_.ndim == 2: # Nx3
                    v_ = v_.reshape(hw + (3,))
                else: # Nx(1+L+M)x3, containing relit renderings
                    v_ = v_.reshape(hw + (v_.shape[1], 3))
            elif k.endswith(('albedo', 'normal')):
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
        # Write images
        img_dict = {}
        alpha = data_dict['gt_alpha']
        for k, v in data_dict.items():
            # RGB
            if k.endswith('rgb'):
                if v.ndim == 3: # HxWx3
                    bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                    img = imgutil.alpha_blend(v, alpha, bg)
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
                else: # HxWx(1+L+M)x3
                    v_orig = v[:, :, 0, :] # lit by original lighting
                    bg = np.ones_like(v_orig) if self.white_bg \
                        else np.zeros_like(v_orig)
                    img = imgutil.alpha_blend(v_orig, alpha, bg)
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
                    # Write relit results
                    novel_light_names = list(self.lights_novel.keys())
                    olat_first_n = np.prod(self.light_res) // 2 # top half only
                    olat_n = 0
                    for i in tqdm(
                            range(1, v.shape[2]), desc="Writing relit results"):
                        lname = novel_light_names[i - 1]
                        # Skip visiualization if enough OLATs
                        if lname.startswith('olat-'):
                            if olat_n >= olat_first_n:
                                continue
                            else:
                                olat_n += 1
                        #
                        k_relit = k + '_' + lname
                        v_relit = v[:, :, i, :]
                        # Compute average lighting
                        lareas = self.lareas.numpy()
                        lareas_upper = lareas[:(lareas.shape[0] // 2), :]
                        weights = np.dstack([lareas_upper] * 3)
                        light = self.lights_novel_uint[lname]
                        light = xm.img.normalize_uint(light) # now float
                        light = xm.img.resize(light, new_h=lareas.shape[0])
                        light_upper = light[:(light.shape[0] // 2), :, :]
                        avg_light = np.average( # (3,)
                            light_upper, axis=(0, 1), weights=weights)
                        # Composite results on average lighting background
                        bg = np.tile(
                            avg_light[None, None, :], v_relit.shape[:2] + (1,))
                        img = xm.img.alpha_blend(v_relit, alpha, bg)
                        img_dict[k_relit] = xm.io.img.write_arr(
                            img, join(outdir, k_relit + '.png'), clip=True)
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
                # If relit results are rendered (e.g., during testing), let's
                # also visualize per-light vis.
                if data_dict['pred_rgb'].ndim == 4:
                    for i in tqdm(
                            range(4 if self.debug else v.shape[2] // 2), # half
                            desc="Writing per-light visibility (%s)" % k):
                        v_olat = v[:, :, i]
                        ij = np.unravel_index(i, self.light_res)
                        k_olat = k + '_olat-%04d-%04d' % ij
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
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            # ioutil.dump_dict_tensors(data_dict, dump_raw_to)
            pass

    @staticmethod
    def _brdf_prop_as_img(brdf_prop):
        """Roughness in microfacet: last dimension is 1, and organically [0, 1].
        """
        rough = brdf_prop[:, :, 0]
        return rough

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train', fps=12):
        self._validate_mode(mode)
        # Shortcircuit if training (same reason as above)
        if mode == 'train':
            return None
        # Validation or testing
        if mode == 'vali':
            outpath = outpref + '.html'
            self._compile_into_webpage(batch_vis_dirs, outpath)
        else:
            outpath = outpref + '.webm'
            self._compile_into_video(batch_vis_dirs, outpath, fps=fps)
        view_at = 'https://viewer' + outpath
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

    def _compile_into_video(self, batch_dirs, out_webm, fps=12):
        batch_dirs = sorted(
            batch_dirs) # assuming batch directory order is the right view order
        if self.debug:
            batch_dirs = batch_dirs[:10]
        # Tonemap and visualize all lighting conditions used
        orig_light_uint = lightutil.vis_light(self.light, h=self.embed_light_h)

        def make_frame(path_dict, light=None, task='viewsyn'):
            # Load predictions
            albedo = xm.io.img.load(path_dict['albedo'])
            lvis = xm.io.img.load(path_dict['lvis'])
            normal = xm.io.img.load(path_dict['normal'])
            rgb = xm.io.img.load(path_dict['rgb'])
            brdf = xm.io.img.load(path_dict['brdf'])
            # Optionally, embed the light used to right bottom corner of render
            if light is not None:
                imgutil.frame_image(light, rgb=(1, 1, 1), width=1)
                rgb[:light.shape[0], -light.shape[1]:] = light
            # Put labels
            hw = rgb.shape[:2]
            put_text_kwargs = {
                'label_top_left_xy': (
                    int(self.put_text_param['text_loc_ratio'] * hw[1]),
                    int(self.put_text_param['text_loc_ratio'] * hw[0])),
                'font_size': int(
                    self.put_text_param['text_size_ratio'] * hw[0]),
                'font_color': (0, 0, 0) if self.white_bg else (1, 1, 1),
                'font_ttf': self.put_text_param['font_path']}
            albedo = xm.vis.text.put_text(albedo, "Albedo", **put_text_kwargs)
            lvis_label = "Light Visibility"
            if task in ('viewsyn', 'simul'):
                lvis_label += " (mean)"
            lvis = xm.vis.text.put_text(lvis, lvis_label, **put_text_kwargs)
            normal = xm.vis.text.put_text(normal, "Normals", **put_text_kwargs)
            rgb = xm.vis.text.put_text(rgb, "Rendering", **put_text_kwargs)
            brdf = xm.vis.text.put_text(brdf, "BRDF", **put_text_kwargs)
            # Make collage
            frame_top = imgutil.hconcat((normal, lvis, brdf))
            frame_bottom = imgutil.hconcat((albedo, rgb))
            frame = imgutil.vconcat((frame_top, frame_bottom))
            return frame

        # ------ View synthesis
        frames = []
        for batch_dir in tqdm(batch_dirs, desc="View synthesis"):
            paths = {
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
        lvis_paths = sorted(
            gfile.Glob(join(relight_view_dir, 'pred_lvis_olat*.png')))
        for lvis_path in tqdm(lvis_paths, desc="Final view, OLAT"):
            olat_id = basename(lvis_path)[len('pred_lvis_'):-len('.png')]
            if self.debug and (olat_id not in self.lights_novel_uint):
                continue
            rgb_path = join(relight_view_dir, 'pred_rgb_%s.png' % olat_id)
            paths = {
                'albedo': join(relight_view_dir, 'pred_albedo.png'),
                'lvis': lvis_path, # per-light
                'normal': join(relight_view_dir, 'pred_normal.png'),
                'rgb': rgb_path,
                'alpha': join(relight_view_dir, 'gt_alpha.png'),
                'brdf': join(relight_view_dir, 'pred_brdf.png')}
            if not ioutil.all_exist(paths):
                logger.warn("Skipping because of missing files: %s", olat_id)
                continue
            light_used = self.lights_novel_uint[olat_id]
            frame = make_frame(paths, light=light_used, task='relight')
            frames.append(frame)
        # ------ Simultaneous
        envmap_names = [
            x for x in self.lights_novel.keys() if not x.startswith('olat')]
        n_envmaps = len(envmap_names)
        batch_dirs_roundtrip = list(reversed(batch_dirs)) + batch_dirs
        n_views_per_envmap = len(batch_dirs_roundtrip) / n_envmaps # float
        map_i = 0
        for view_i, batch_dir in enumerate(
                tqdm(batch_dirs_roundtrip, desc="View roundtrip, IBL")):
            envmap_name = envmap_names[map_i]
            rgb_path = join(batch_dir, 'pred_rgb_%s.png' % envmap_name)
            paths = {
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
            light_used = self.lights_novel_uint[envmap_name]
            frame = make_frame(paths, light=light_used, task='simul')
            frames.append(frame)
            # Time to switch to the next map?
            if (view_i + 1) > n_views_per_envmap * (map_i + 1):
                map_i += 1
        #
        ioutil.write_video(frames, out_webm, fps=fps)
