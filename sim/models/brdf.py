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
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.brdf.renderer import SphereRenderer
from google3.experimental.users.xiuming.sim.brdf.merl.merl import MERL

from .base import Model as BaseModel
from ..networks import mlp
from ..networks.embedder import Embedder
from ..networks.layers import LatentCode
from ..util import logging as logutil, io as ioutil


logger = logutil.Logger(loggee="models/brdf")


class Model(BaseModel):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)
        self.mlp_chunk = self.config.getint('DEFAULT', 'mlp_chunk')
        # Embedders
        self.embedder = self._init_embedder()
        # Network components
        self.net = self._init_net()
        # Get BRDF names
        data_dir = self.config.get('DEFAULT', 'data_root')
        train_npz = sorted(gfile.Glob(join(data_dir, 'train_*.npz')))
        self.brdf_names = [
            basename(x)[len('train_'):-len('.npz')] for x in train_npz]
        # Add latent codes to optimize so that they get registered as trainable
        z_dim = self.config.getint('DEFAULT', 'z_dim')
        z_gauss_mean = self.config.getfloat('DEFAULT', 'z_gauss_mean')
        z_gauss_std = self.config.getfloat('DEFAULT', 'z_gauss_std')
        normalize_z = self.config.getboolean('DEFAULT', 'normalize_z')
        n_brdfs = len(self.brdf_names)
        self.latent_code = LatentCode(
            n_brdfs, z_dim, mean=z_gauss_mean, std=z_gauss_std,
            normalize=normalize_z)

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net = {}
        net['brdf_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['brdf_out'] = mlp.Network([1], act=['softplus']) # > 0
        return net

    def _init_embedder(self):
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        n_freqs = self.config.getint('DEFAULT', 'n_freqs')
        # Shortcircuit if not using embedders
        if not pos_enc:
            embedder = {'rusink': tf.identity}
            return embedder
        # Rusink. coordinate embedder
        kwargs = {
            'incl_input': True,
            'in_dims': 3,
            'log2_max_freq': n_freqs - 1,
            'n_freqs': n_freqs,
            'log_sampling': True,
            'periodic_func': [tf.math.sin, tf.math.cos]}
        embedder_rusink = Embedder(**kwargs)
        embedder = {'rusink': embedder_rusink}
        return embedder

    def call(self, batch, mode='train'):
        self._validate_mode(mode)
        id_, i, envmap_h, ims, spp, rusink, refl = batch
        if mode == 'test' and i[0] == -1:
            # Novel identities -- need interpolation
            i_w1_mat1_w2_mat2 = id_[0].numpy().decode()
            _, w1, mat1, w2, mat2 = i_w1_mat1_w2_mat2.split('_')
            w1, w2 = float(w1), float(w2)
            i1, i2 = self.brdf_names.index(mat1), self.brdf_names.index(mat2)
            z = self.latent_code.interp(w1, i1, w2, i2)
            z = tf.tile(z, (id_.shape[0], 1))
        else:
            z = self.latent_code(i)
        brdf, brdf_reci = self._eval_brdf_at(z, rusink)
        # For loss computation
        pred = {'brdf': brdf, 'brdf_reci': brdf_reci}
        gt = {'brdf': refl}
        loss_kwargs = {}
        # To visualize
        to_vis = {
            'id': id_, 'i': i, 'z': z, 'gt_brdf': refl,
            'envmap_h': envmap_h, 'ims': ims, 'spp': spp}
        for k, v in pred.items():
            to_vis[k] = v
        return pred, gt, loss_kwargs, to_vis

    def _eval_brdf_at(self, z, rusink):
        mlp_layers = self.net['brdf_mlp']
        out_layer = self.net['brdf_out']
        # Chunk by chunk to avoid OOM
        chunks, chunks_reci = [], []
        iterator = range(0, rusink.shape[0], self.mlp_chunk)
        if len(iterator) > 1:
            iterator = tqdm(iterator, desc="MLP Chunks")
        for i in iterator:
            end_i = min(rusink.shape[0], i + self.mlp_chunk)
            z_chunk = z[i:end_i]
            rusink_chunk = rusink[i:end_i, :]
            rusink_embed = self.embedder['rusink'](rusink_chunk)
            z_rusink = tf.concat((z_chunk, rusink_embed), axis=1)
            chunk = out_layer(mlp_layers(z_rusink))
            chunks.append(chunk)
            # Reciprocity
            phid = rusink[i:end_i, :1]
            thetah_thetad = rusink[i:end_i, 1:]
            rusink_chunk = tf.concat((phid + np.pi, thetah_thetad), axis=1)
            rusink_embed = self.embedder['rusink'](rusink_chunk)
            z_rusink = tf.concat((z_chunk, rusink_embed), axis=1)
            chunk = out_layer(mlp_layers(z_rusink))
            chunks_reci.append(chunk)
        brdf = tf.concat(chunks, axis=0)
        brdf_reci = tf.concat(chunks_reci, axis=0)
        return brdf, brdf_reci # (n_rusink, 1)

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
        # Accumulate loss
        loss = 0
        for weight, loss_func in self.wloss:
            loss += weight * loss_func(f(gt['brdf']), f(pred['brdf']), **kwargs)
            # Same ground truth for the reciprocal Rusink.
            loss += weight * loss_func(
                f(gt['brdf']), f(pred['brdf_reci']), **kwargs)
        return loss

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None, n_vis=64):
        self._validate_mode(mode)
        # Shortcircuit if training
        if mode == 'train':
            return
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            # ioutil.dump_dict_tensors(data_dict, dump_raw_to)
            pass
        # "Visualize" metadata
        id_ = data_dict['id'][0]
        id_ = id_.numpy().decode()
        metadata_out = join(outdir, 'metadata.json')
        metadata = {'id': id_}
        ioutil.write_json(metadata, metadata_out)
        # Visualize the latent codes
        z = data_dict['z'][0, :].numpy()
        z_png = join(outdir, 'z.png')
        plot = xm.vis.plot.Plot(outpath=z_png)
        plot.bar(z)
        # Visualize the BRDF values
        pred = data_dict['brdf'].numpy()
        pred_reci = data_dict['brdf_reci'].numpy()
        brdf_val = np.hstack((pred_reci, pred))
        labels = ['Pred. (reci.)', 'Pred.']
        if mode == 'vali':
            gt = data_dict['gt_brdf'].numpy()
            brdf_val = np.hstack((brdf_val, gt))
            labels.append('GT')
        brdf_val = brdf_val[::int(brdf_val.shape[0] / n_vis), :] # just a subset
        brdf_val = np.log10(brdf_val) # log scale
        brdf_png = join(outdir, 'log10_brdf.png')
        plot = xm.vis.plot.Plot(labels=labels, outpath=brdf_png)
        plot.bar(brdf_val)
        if mode == 'vali':
            return
        # If testing, continue to visualize characteristic slice
        merl = MERL()
        envmap_h = data_dict['envmap_h'][0].numpy()
        ims = data_dict['ims'][0].numpy()
        spp = data_dict['spp'][0].numpy()
        renderer = SphereRenderer(
            'point', outdir, envmap_h=envmap_h, envmap_inten=40, ims=ims,
            spp=spp)
        cslice_out = join(outdir, 'cslice.png')
        cslice_shape = merl.cube_rusink.shape[1:]
        cslice_end_i = np.prod(cslice_shape[:2])
        pred_cslice = pred[:cslice_end_i, :] # first 90x90 are for char. slices
        cslice = pred_cslice.reshape(cslice_shape[:2])
        cslice_img = merl.characteristic_slice_as_img(cslice)
        xm.io.img.write_img(cslice_img, cslice_out)
        # ... and render the predicted BRDF
        render_out = join(outdir, 'render.png')
        pred_render = pred[cslice_end_i:, :] # remaining are for rendering
        brdf = np.zeros_like(renderer.lcontrib)
        brdf[renderer.lvis.astype(bool)] = pred_render
        render = renderer.render(brdf)
        xm.io.img.write_arr(render, render_out, clip=True)

    def compile_batch_vis(
            self, batch_vis_dirs, outpref, mode='train', marker_size=16, fps=2):
        """If in 'test' mode, compiles visualzied results into:
            (1) An HTML of reconstructions of seen identities; and
            (2) A video of interpolating between seen identities.
        """
        vis_dir = '/data/brdf/merl_sep_npz/ims512_envmaph16_spp1/'#self.config.get('DEFAULT', 'data_root')#FIXME
        font_path = self.config.get('DEFAULT', 'font_path')
        self._validate_mode(mode)
        viewer_http = 'https://viewer'
        # Shortcircuit if training
        if mode == 'train':
            return None
        # Put optimized latent code and BRDF value visualizations into HTML
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_vis_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            id_ = metadata['id']
            metadata = str(metadata)
            row = [
                metadata,
                join(batch_dir, 'z.png'),
                join(batch_dir, 'log10_brdf.png')]
            rowcaps = ["Metadata", "Latent Code", "BRDF (log-scale)"]
            rowtypes = ['text', 'image', 'image']
            # If we are testing, additional columns for char. slices and renders
            if mode == 'test':
                pred_cslice_path = join(batch_dir, 'cslice.png')
                pred_render_path = join(batch_dir, 'render.png')
                if '_' in id_:
                    # Interpolated identities
                    row_extra = [
                        pred_cslice_path, pred_render_path, "N/A", "N/A"]
                    rowtypes_extra = ['image', 'image', 'text', 'text']
                else:
                    # Seen identities
                    gt_cslice_path = join(vis_dir, 'vis-cslice_%s.png' % id_)
                    gt_render_path = join(vis_dir, 'vis-render_%s.png' % id_)
                    row_extra = [
                        pred_cslice_path, pred_render_path, gt_cslice_path,
                        gt_render_path]
                    rowtypes_extra = ['image', 'image', 'image', 'image']
                row += row_extra
                rowcaps += [
                    "Pred. (char. slice)", "Pred. (render)", "GT (char. slice)",
                    "GT (render)"]
                rowtypes += rowtypes_extra
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Make HTML
        html = xm.vis.html.HTML(bgcolor='white', text_color='black')
        html.add_header("BRDF-MLP")
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        # Write HTML
        out_html = outpref + '.html'
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)
        view_at = viewer_http + out_html
        # Done if validation
        if mode == 'vali':
            return view_at
        # Testing, so continue to make a video for interpolation
        frame_ind, frames = [], []
        for batch_dir in batch_vis_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            id_ = metadata['id']
            # Skip if this is a seen identity
            if '_' not in id_:
                continue
            i, w1, mat1_id, w2, mat2_id = id_.split('_')
            i = int(i)
            w1, w2 = float(w1), float(w2)
            mat1_path = join(vis_dir, 'vis-render_%s.png' % mat1_id)
            mat2_path = join(vis_dir, 'vis-render_%s.png' % mat2_id)
            pred_path = join(batch_dir, 'render.png')
            mat1 = xm.io.img.load(mat1_path)
            mat2 = xm.io.img.load(mat2_path)
            pred = xm.io.img.load(pred_path)
            # Resize according to width because we will vertically concat.
            mat1 = xm.img.resize(mat1, new_w=pred.shape[1])
            mat2 = xm.img.resize(mat2, new_w=pred.shape[1])
            # Label the maps
            font_size = int(0.06 * pred.shape[1])
            label_kwargs = {
                'font_color': (0, 0, 0), 'font_ttf': font_path,
                'font_size': font_size}
            mat1_labeled = xm.vis.text.put_text(mat1, "Mat. 1", **label_kwargs)
            mat2_labeled = xm.vis.text.put_text(mat2, "Mat. 2", **label_kwargs)
            marker_i = int(w2 * pred.shape[0])
            marker_vstart = max(0, marker_i - marker_size // 2)
            marker_vend = min(marker_i + marker_size // 2, pred.shape[0] - 1)
            maxv = np.iinfo(pred.dtype).max
            red = np.array((maxv, 0, 0)).reshape((1, 1, 3))
            pred[marker_vstart:marker_vend, :marker_size, :] = red
            frame = np.vstack((mat1_labeled, pred, mat2_labeled))
            frames.append(frame)
            frame_ind.append(i)
        out_webm = outpref + '.webm'
        frames_sort = [
            y for (x, y) in sorted(
                zip(frame_ind, frames), key=lambda pair: pair[0])]
        ioutil.write_video(frames_sort, out_webm, fps=fps)
        view_at += '\n\t%s' % (viewer_http + out_webm)
        return view_at # to be logged into TensorBoard
