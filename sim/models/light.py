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
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

from .brdf import Model as BaseModel
from ..networks import mlp
from ..networks.embedder import Embedder
from ..util import logging as logutil, io as ioutil, light as lightutil


logger = logutil.Logger(loggee="models/light")


class Model(BaseModel):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)
        self.light_names = self.brdf_names
        delattr(self, 'brdf_names')

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net = {}
        net['light_mlp'] = mlp.Network(
            [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['light_out'] = mlp.Network([3], act=['softplus']) # > 0
        return net

    def _init_embedder(self):
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        n_freqs = self.config.getint('DEFAULT', 'n_freqs')
        # Shortcircuit if not using embedders
        if not pos_enc:
            embedder = {'latlng': tf.identity}
            return embedder
        # Lat.-long. coordinate embedder
        kwargs = {
            'incl_input': True,
            'in_dims': 2,
            'log2_max_freq': n_freqs - 1,
            'n_freqs': n_freqs,
            'log_sampling': True,
            'periodic_func': [tf.math.sin, tf.math.cos]}
        embedder_latlng = Embedder(**kwargs)
        embedder = {'latlng': embedder_latlng}
        return embedder

    def call(self, batch, mode='train'):
        self._validate_mode(mode)
        id_, i, envmap_h, latlng, radi = batch
        if mode == 'test' and i[0] == -1:
            # Novel identities -- need interpolation
            i_w1_l1_w2_l2 = id_[0].numpy().decode()
            _, w1, l1, w2, l2 = i_w1_l1_w2_l2.split('_')
            w1, w2 = float(w1), float(w2)
            i1, i2 = self.light_names.index(l1), self.light_names.index(l2)
            z = self.latent_code.interp(w1, i1, w2, i2)
            z = tf.tile(z, (id_.shape[0], 1))
        else:
            z = self.latent_code(i)
        pred = self.eval_light_at(z, latlng)
        # For loss computation
        pred = {'light': pred}
        gt = {'light': radi}
        loss_kwargs = {}
        # To visualize
        to_vis = {
            'id': id_, 'i': i, 'z': z, 'gt_light': radi,
            'envmap_h': envmap_h}
        for k, v in pred.items():
            to_vis[k] = v
        return pred, gt, loss_kwargs, to_vis

    def eval_light_at(self, z, latlng):
        mlp_layers = self.net['light_mlp']
        out_layer = self.net['light_out']
        # Chunk by chunk to avoid OOM
        chunks = [ # always have an empty tensor to avoid tf.concat([], axis=0),
            # which could happen when there are too few validation points
            tf.reshape(tf.convert_to_tensor(()), (0, 3))]
        iterator = range(0, latlng.shape[0], self.mlp_chunk)
        if len(iterator) > 1:
            iterator = tqdm(iterator, desc="MLP Chunks")
        for i in iterator:
            end_i = min(latlng.shape[0], i + self.mlp_chunk)
            z_chunk = z[i:end_i]
            latlng_chunk = latlng[i:end_i, :]
            latlng_embed = self.embedder['latlng'](latlng_chunk)
            z_rusink = tf.concat((z_chunk, latlng_embed), axis=1)
            chunk = out_layer(mlp_layers(z_rusink))
            chunks.append(chunk)
        radi = tf.concat(chunks, axis=0)
        return radi # (n_rusink, 3)

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
            radi_loss = loss_func(f(gt['light']), f(pred['light']), **kwargs)
            loss += weight * radi_loss
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            vis_envmap_h=256):
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
        # Visualize the radiance values
        pred = data_dict['light'].numpy()
        light_val = pred
        labels = ['Pred. (R)', 'Pred. (G)', 'Pred. (B)']
        if mode == 'vali':
            gt = data_dict['gt_light'].numpy()
            light_val = np.hstack((light_val, gt))
            labels += ['GT (R)', 'GT (G)', 'GT (B)']
        light_val = np.log10(light_val) # log scale
        light_png = join(outdir, 'log10_light.png')
        plot = xm.vis.plot.Plot(labels=labels, outpath=light_png)
        plot.bar(light_val)
        if mode == 'vali':
            return
        # If testing, continue to visualize the entire environment map
        envmap_h = data_dict['envmap_h'][0].numpy()
        hdr = np.reshape(pred, (envmap_h, 2 * envmap_h, 3))
        outpath = join(outdir, 'envmap.png')
        lightutil.vis_light(hdr, outpath=outpath, h=vis_envmap_h)

    def compile_batch_vis(
            self, batch_vis_dirs, outpref, mode='train', marker_size=16, fps=2):
        """If in 'test' mode, compiles visualzied results into:
            (1) An HTML of reconstructions of seen identities; and
            (2) A video of interpolating between seen identities.
        """
        vis_dir = self.config.get('DEFAULT', 'data_root')
        font_path = self.config.get('DEFAULT', 'font_path')
        self._validate_mode(mode)
        viewer_http = 'https://viewer'
        # No-op if training
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
                join(batch_dir, 'log10_light.png')]
            rowcaps = ["Metadata", "Latent Code", "Light (log-scale)"]
            rowtypes = ['text', 'image', 'image']
            # If we are testing, additional columns for full maps
            if mode == 'test':
                pred_path = join(batch_dir, 'envmap.png')
                if '_' in id_:
                    # Interpolated identities
                    row_extra = [pred_path, "N/A"]
                    rowtypes_extra = ['image', 'text']
                else:
                    # Seen identities
                    gt_path = join(vis_dir, 'vis_%s.png' % id_)
                    row_extra = [pred_path, gt_path]
                    rowtypes_extra = ['image', 'image']
                row += row_extra
                rowcaps += ["Pred.", "GT"]
                rowtypes += rowtypes_extra
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Make HTML
        html = xm.vis.html.HTML(bgcolor='white', text_color='black')
        html.add_header("Light-MLP")
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
            i, w1, map1_id, w2, map2_id = id_.split('_')
            i = int(i)
            w1, w2 = float(w1), float(w2)
            map1_path = join(vis_dir, 'vis_%s.png' % map1_id)
            map2_path = join(vis_dir, 'vis_%s.png' % map2_id)
            pred_path = join(batch_dir, 'envmap.png')
            map1 = xm.io.img.load(map1_path)
            map2 = xm.io.img.load(map2_path)
            pred = xm.io.img.load(pred_path)
            # Resize according to width because we will vertically concat.
            map1 = xm.img.resize(map1, new_w=pred.shape[1])
            map2 = xm.img.resize(map2, new_w=pred.shape[1])
            # Label the maps
            label_kwargs = {'font_color': (0, 0, 0), 'font_ttf': font_path}
            map1_labeled = xm.vis.text.put_text(map1, "Map 1", **label_kwargs)
            map2_labeled = xm.vis.text.put_text(map2, "Map 2", **label_kwargs)
            marker_i = int(w2 * pred.shape[0])
            marker_vstart = max(0, marker_i - marker_size // 2)
            marker_vend = min(marker_i + marker_size // 2, pred.shape[0] - 1)
            maxv = np.iinfo(pred.dtype).max
            red = np.array((maxv, 0, 0)).reshape((1, 1, 3))
            pred[marker_vstart:marker_vend, :marker_size, :] = red
            frame = np.vstack((map1_labeled, pred, map2_labeled))
            frames.append(frame)
            frame_ind.append(i)
        out_webm = outpref + '.webm'
        frames_sort = [
            y for (x, y) in sorted(
                zip(frame_ind, frames), key=lambda pair: pair[0])]
        ioutil.write_video(frames_sort, out_webm, fps=fps)
        view_at += '\n\t%s' % (viewer_http + out_webm)
        return view_at # to be logged into TensorBoard
