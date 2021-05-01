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

from os.path import join
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.networks import mlp
from nerfactor.networks.embedder import Embedder
from nerfactor.util import logging as logutil, io as ioutil, img as imgutil, \
    tensor as tutil, math as mathutil
from nerfactor.models.base import Model as BaseModel


logger = logutil.Logger(loggee="models/shape")


class Model(BaseModel):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)
        self.white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        self.mlp_chunk = self.config.getint('DEFAULT', 'mlp_chunk')
        # ------ Shape
        self.normal_smooth_weight = self.config.getfloat(
            'DEFAULT', 'normal_smooth_weight', fallback=0.)
        self.lvis_smooth_weight = self.config.getfloat(
            'DEFAULT', 'lvis_smooth_weight', fallback=0.)
        # ------ Embedders
        self.embedder = self._init_embedder()
        # ------ Network components
        self.net = self._init_net()
        # ------ Lighting
        light_h = self.config.getint('DEFAULT', 'light_h')
        light_w = int(2 * light_h)
        lxyz, _ = gen_light_xyz(light_h, light_w)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        #
        self.put_text_param = {
            'text_loc_ratio': 0.05, 'text_size_ratio': 0.05,
            'font_path': xm.const.Path.open_sans_regular}

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net = {}
        # Normals
        net['normal_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['normal_out'] = mlp.Network([3], act=None) # normalized elsewhere
        # Light visibility
        net['lvis_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['lvis_out'] = mlp.Network([1], act=['sigmoid']) # [0, 1]
        return net

    def _init_embedder(self):
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        n_freqs_xyz = self.config.getint('DEFAULT', 'n_freqs_xyz')
        n_freqs_ldir = self.config.getint('DEFAULT', 'n_freqs_ldir')
        n_freqs_vdir = self.config.getint('DEFAULT', 'n_freqs_vdir')
        # Shortcircuit if not using embedders
        if not pos_enc:
            embedder = {
                'xyz': tf.identity, 'ldir': tf.identity, 'vdir': tf.identity}
            return embedder
        # Position embedder
        kwargs = {
            'incl_input': True,
            'in_dims': 3,
            'log2_max_freq': n_freqs_xyz - 1,
            'n_freqs': n_freqs_xyz,
            'log_sampling': True,
            'periodic_func': [tf.math.sin, tf.math.cos]}
        embedder_xyz = Embedder(**kwargs)
        # Light direction embedder
        kwargs['log2_max_freq'] = n_freqs_ldir - 1
        kwargs['n_freqs'] = n_freqs_ldir
        embedder_ldir = Embedder(**kwargs)
        # View direction embedder
        kwargs['log2_max_freq'] = n_freqs_vdir - 1
        kwargs['n_freqs'] = n_freqs_vdir
        embedder_vdir = Embedder(**kwargs)
        #
        embedder = {
            'xyz': embedder_xyz, 'ldir': embedder_ldir, 'vdir': embedder_vdir}
        return embedder

    def _calc_ldir(self, pts):
        surf2l = tf.reshape(
            self.lxyz, (1, -1, 3)) - pts[:, None, :]
        surf2l = mathutil.safe_l2_normalize(surf2l, axis=2)
        tf.debugging.assert_greater(
            tf.linalg.norm(surf2l, axis=2), 0.,
            message="Found zero-norm light directions")
        return surf2l # NxLx3

    @staticmethod
    def _calc_vdir(cam_loc, pts):
        surf2c = cam_loc - pts
        surf2c = mathutil.safe_l2_normalize(surf2c, axis=1)
        tf.debugging.assert_greater(
            tf.linalg.norm(surf2c, axis=1), 0.,
            message="Found zero-norm view directions")
        return surf2c # Nx3

    def call(self, batch, mode='train'):
        xyz_jitter_std = self.config.getfloat('DEFAULT', 'xyz_jitter_std')
        self._validate_mode(mode)
        id_, hw, _, _, _, alpha, xyz, normal, lvis = batch
        surf2l = self._calc_ldir(xyz)
        # Jitter XYZs
        if xyz_jitter_std > 0:
            xyz_noise = tf.random.normal(tf.shape(xyz), stddev=xyz_jitter_std)
        else:
            xyz_noise = None
        # ------ Normals
        normal_pred = self._pred_normal_at(xyz)
        if xyz_noise is not None and self.normal_smooth_weight > 0:
            normal_jitter = self._pred_normal_at(xyz + xyz_noise)
        else:
            normal_jitter = None
        normal_pred = mathutil.safe_l2_normalize(normal_pred, axis=1)
        if normal_jitter is not None:
            normal_jitter = mathutil.safe_l2_normalize(normal_jitter, axis=1)
        # ------ Light visibility
        lvis_pred = self._pred_lvis_at(xyz, surf2l)
        if xyz_noise is not None and self.lvis_smooth_weight > 0:
            lvis_jitter = self._pred_lvis_at(xyz + xyz_noise, surf2l)
        else:
            lvis_jitter = None
        # ------ Loss
        pred = {'normal': normal_pred, 'lvis': lvis_pred}
        gt = {'normal': normal, 'lvis': lvis, 'alpha': alpha}
        loss_kwargs = {
            'normal_jitter': normal_jitter, 'lvis_jitter': lvis_jitter}
        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    @staticmethod
    def chunk_apply(func, x, dim, chunk_size):
        n = tf.shape(x)[0]
        y = tf.zeros((n, dim), dtype=tf.float32)
        for i in tf.range(0, n, chunk_size):
            end_i = tf.math.minimum(n, i + chunk_size)
            x_chunk = x[i:end_i]
            y_chunk = func(x_chunk)
            y = tf.tensor_scatter_nd_update(
                y, tf.range(i, end_i)[:, None], y_chunk)
        return y

    def _pred_normal_at(self, pts, eps=1e-6):
        mlp_layers = self.net['normal_mlp']
        out_layer = self.net['normal_out']

        def chunk_func(surf):
            surf_embed = self.embedder['xyz'](surf)
            normals = out_layer(mlp_layers(surf_embed))
            return normals

        normal = self.chunk_apply(chunk_func, pts, 3, self.mlp_chunk)
        normal += eps # to avoid all-zero normals messing up tangents
        tf.debugging.assert_greater(
            tf.linalg.norm(normal, axis=1), 0.,
            message="Found zero-norm normals")
        return normal # Nx3

    def _pred_lvis_at(self, pts, surf2l):
        mlp_layers = self.net['lvis_mlp']
        out_layer = self.net['lvis_out']

        # Flattening
        n_lights = tf.shape(surf2l)[1]
        surf2l_flat = tf.reshape(surf2l, (-1, 3)) # NLx3
        # Repeating surface points to match light directions
        surf = tf.tile(pts[:, None, :], (1, n_lights, 1)) # NxLx3
        surf_flat = tf.reshape(surf, (-1, 3)) # NLx3

        def chunk_func(surf_surf2l):
            surf, surf2l = surf_surf2l[:, :3], surf_surf2l[:, 3:]
            surf_embed = self.embedder['xyz'](surf)
            surf2l_embed = self.embedder['ldir'](surf2l)
            surf_surf2l = tf.concat((surf_embed, surf2l_embed), -1)
            lvis = out_layer(mlp_layers(surf_surf2l))
            return lvis

        surf_surf2l = tf.concat((surf_flat, surf2l_flat), 1) # NLx6
        lvis_flat = self.chunk_apply(chunk_func, surf_surf2l, 1, self.mlp_chunk)
        lvis = tf.reshape(lvis_flat, (tf.shape(pts)[0], n_lights)) # NxL
        lvis = tf.debugging.check_numerics(lvis, "Light visibility")
        return lvis # NxL

    def compute_loss(self, pred, gt, **kwargs):
        """Composites signals onto white backgrounds and computes the loss.
        """
        normal_loss_weight = self.config.getfloat(
            'DEFAULT', 'normal_loss_weight')
        lvis_loss_weight = self.config.getfloat('DEFAULT', 'lvis_loss_weight')
        smooth_use_l1 = self.config.getboolean('DEFAULT', 'smooth_use_l1')
        smooth_loss = tf.keras.losses.MAE if smooth_use_l1 \
            else tf.keras.losses.MSE
        #
        normal_jitter = kwargs.pop('normal_jitter')
        lvis_jitter = kwargs.pop('lvis_jitter')
        #
        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
        # Composite predictions and ground truth onto backgrounds
        alpha = gt['alpha']
        bg = tf.ones_like(normal_gt) if self.white_bg \
            else tf.zeros_like(normal_gt)
        normal_pred = imgutil.alpha_blend(normal_pred, alpha, tensor2=bg)
        normal_gt = imgutil.alpha_blend(normal_gt, alpha, tensor2=bg)
        bg = tf.ones_like(lvis_gt) if self.white_bg else tf.zeros_like(lvis_gt)
        lvis_pred = imgutil.alpha_blend(lvis_pred, alpha, tensor2=bg)
        lvis_gt = imgutil.alpha_blend(lvis_gt, alpha, tensor2=bg)
        # Predicted values should be close to NeRF values
        loss = 0
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
            if k.endswith('normal'):
                v_ = v_.reshape(hw + (3,))
            elif k.endswith(('occu', 'alpha')):
                v_ = v_.reshape(hw)
            elif k.endswith('lvis'):
                v_ = v_.reshape(hw + (v_.shape[1],))
            else:
                raise NotImplementedError(k)
            data_dict[k] = v_
        # Write images
        img_dict = {}
        alpha = data_dict['gt_alpha']
        for k, v in data_dict.items():
            if k.endswith('normal'):
                v = (v + 1) / 2 # [-1, 1] to [0, 1]
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif k.endswith('lvis'):
                v = np.mean(v, axis=2) # NOTE: average across all lights
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('occu', 'alpha')):
                img_dict[k] = xm.io.img.write_arr(
                    v, join(outdir, k + '.png'), clip=True)
            else:
                raise NotImplementedError(k)
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
        metadata = {'id': id_}
        ioutil.write_json(metadata, join(outdir, 'metadata.json'))
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            # ioutil.dump_dict_tensors(data_dict, dump_raw_to)
            pass

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train'):
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
            raise NotImplementedError(mode)
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
                join(batch_dir, 'pred-vs-gt_normal.apng'),
                join(batch_dir, 'pred-vs-gt_lvis.apng')]
            rowcaps = ["Metadata", "Normal", "Light Visibility"]
            rowtypes = ['text', 'image', 'image']
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
        header = "Refining and Caching NeRF Geometry"
        html.add_header(header)
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)
