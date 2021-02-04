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

from os.path import join
import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import plotly.graph_objects as go

from absl import app
from google3.pyglib import gfile
from google3.experimental.users.xiuming.sim.sim import datasets, models
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil


def query_opacity(model, pts, voxel_size, use_fine=False, mlp_chunk=65536):
    pref = 'fine_' if use_fine else 'coarse_'
    enc = model.net[pref + 'enc']
    a_out = model.net[pref + 'a_out']
    embedder = model.embedder['xyz']
    pts_flat = tf.reshape(pts, (-1, 3))
    # Chunk by chunk to avoid OOM
    a_chunks = []
    for i in range(0, pts_flat.shape[0], mlp_chunk):
        pts_chunk = pts_flat[i:min(pts_flat.shape[0], i + mlp_chunk)]
        pts_embed = embedder(pts_chunk)
        feat = enc(pts_embed)
        a_flat = a_out(feat)
        a_chunks.append(a_flat)
    a_flat = tf.concat(a_chunks, axis=0)
    opacity_flat = 1.0 - tf.exp(-tf.nn.relu(a_flat) * voxel_size)
    opacity = tf.reshape(opacity_flat, pts.shape[:3])
    return opacity


def gen_para_rays(near, far, res):
    """Ensures the volume is square, and all voxels are of the same size.
    """
    d = far - near
    # go/holodeck-output-api#cameras
    x = tf.linspace(-d / 2, d / 2, res) # +X is from right to left arm
    y = tf.linspace(0., d, res) # +Y is from feet to head
    z = tf.linspace(near, far, res) # +Z is from back to chest
    xyz = tf.stack(tf.meshgrid(x, y, z, indexing='xy'), axis=-1)
    return xyz


def main(_):
    ckpt_path = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/nerf_repro/324_20190806_134352_viewsyn_0235_transp-bg/lr:0.0001|mgm:-1/vis_test/ckpt-169'
    res = 32

    config_ini = get_config_ini(ckpt_path)
    config = ioutil.read_config(config_ini)

    # Make dataset
    datapipe = make_datapipe(config)

    # Restore model
    model = restore_model(config, ckpt_path)
    posenc = model.embedder['xyz']
    enc = model.net['fine_enc']
    a_out = model.net['fine_a_out']

    # Generate rays starting at the XY plane, parallel to the Z axis
    near = config.getfloat('DEFAULT', 'near')
    far = config.getfloat('DEFAULT', 'far')
    xyz = gen_para_rays(near, far, res)
    voxel_size = (far - near) / res

    # Compute alpha (probability of being absorbed) at each location
    mlp_chunk = config.getint('DEFAULT', 'mlp_chunk')
    with tf.GradientTape() as tape:
        opacity = query_opacity(model, xyz, voxel_size, use_fine=True, mlp_chunk=mlp_chunk)

    # Plot
    xyz = xyz.numpy()
    opacity = opacity.numpy()
    fig = go.Figure(data=go.Volume(
        x=xyz[:, :, :, 0].flatten(), y=xyz[:, :, :, 1].flatten(), z=xyz[:, :, :, 2].flatten(),
        value=opacity.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    from IPython import embed; embed()

    # Run inference on a single batch
    for batch in datapipe.take(1):
        id_, hw, rayo, rayd, rgb, alpha = batch
        from IPython import embed; embed()
        pred, gt, loss_kwargs, to_vis = model.call(batch, mode='vali')

    
def get_config_ini(ckpt_path):
    return '/'.join(ckpt_path.split('/')[:-2]) + '.ini'


def make_datapipe(config):
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'vali')

    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch)
    return datapipe


def restore_model(config, ckpt_path):
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config)

    model.register_trainable()

    # Resume from checkpoint
    assert model.trainable_registered, (
        "Register the trainable layers to have them restored from the "
        "checkpoint")
    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(ckpt_path).expect_partial()

    return model


if __name__ == '__main__':
    app.run(main)
