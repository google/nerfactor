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

from os.path import join, basename, dirname
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from absl import app, flags

from google3.pyglib import gfile
from google3.experimental.users.xiuming.sim.sim import datasets, models
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil, math as mathutil, geom as geomutil
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_string('data_mode', '', "")
flags.DEFINE_string('nerf_data_dir', '', "")
flags.DEFINE_string('albedo_path', '', "")
flags.DEFINE_string('out_dir', '', "")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="microfacet")


def main(_):
    config_ini = get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    nerf_data_dir = FLAGS.nerf_data_dir.rstrip('/')
    desired_id = basename(FLAGS.nerf_data_dir)
    config.set('DEFAULT', 'data_nerf_root', dirname(FLAGS.nerf_data_dir))

    # NOTE: overriding
    config.set('DEFAULT','white_bg','False')
    config.set('DEFAULT','light_h','2')
    config.set('DEFAULT','imh','512')

    #mlp_chunk = config.getint('DEFAULT', 'mlp_chunk')
    #n_samples_coarse = config.getint('DEFAULT', 'n_samples_coarse')
    #n_samples_fine = config.getint('DEFAULT', 'n_samples_fine')
    #lin_in_disp = config.getboolean('DEFAULT', 'lin_in_disp')
    #perturb = config.getboolean('DEFAULT', 'perturb')
    #near = config.getfloat('DEFAULT', 'near')
    #far = config.getfloat('DEFAULT', 'far')
    #light_exr = config.get('DEFAULT', 'gt_light')

    # Make dataset
    datapipe = make_datapipe(config, FLAGS.data_mode)

    # Restore model
    model = restore_model(config, FLAGS.ckpt)
    #coarse_enc = model.net['coarse_enc']
    #coarse_a_out = model.net['coarse_a_out']
    #fine_enc = model.net['fine_enc']
    #fine_a_out = model.net['fine_a_out']
    #embedder = model.embedder['xyz']

    # Run inference on a single batch
    for batch in datapipe:
        id_, hw, rayo, rayd, _, alpha, xyz, normal, lvis = batch

        lvis = tf.clip_by_value(lvis, 0, 1)

        if id_[0].numpy().decode() != desired_id:
            continue

        if not gfile.Exists(FLAGS.out_dir):
            gfile.MakeDirs(FLAGS.out_dir)

        hw = tuple(hw[0, :].numpy())

        lvis_np = lvis.numpy()
        n_lights = lvis_np.shape[1]
        lvis_np = np.sum(lvis_np, axis=1)
        lvis_np = np.reshape(lvis_np, hw)
        lvis_np /= n_lights
        xm.io.img.write_arr(lvis_np, join(FLAGS.out_dir, 'lvis_avg.png'), clip=True)

        # Load ground-truth albedo
        albedo = xm.io.img.load(FLAGS.albedo_path)
        albedo = xm.img.normalize_uint(albedo)
        albedo = xm.img.resize(albedo, new_h=hw[0], new_w=hw[1])
        albedo = np.reshape(albedo, (-1, 3))
        albedo = tf.convert_to_tensor(albedo.astype(np.float32))

        # BRDFs for these surface points
        surf2l = tf.reshape(
            model.lxyz, (1, -1, 3)) - xyz[:, None, :] # (n_surf_pts, n_lights, 3)
        surf2l = tf.math.l2_normalize(surf2l, axis=2)
        surf2c = rayo - xyz # (n_rays, 3)
        surf2c = tf.math.l2_normalize(surf2c, axis=1)
        brdf = model._eval_brdf_at( # (n_surf_pts, n_lights, 3)
            surf2l, surf2c, normal, albedo, glossy_prob=0.2) # NOTE

        brdf_np = brdf.numpy()
        frames = []
        for light_i in range(brdf_np.shape[1]):
            frame = brdf[:, light_i, :]
            frame = np.reshape(frame, hw + (3,))
            frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        brdf_vis = join(FLAGS.out_dir, 'brdf.webm')
        ioutil.write_video(frames, brdf_vis)

        # Rendering equation
        rgb = model._eval_render_eqn(
            lvis, brdf, surf2l, normal, white_light_override=True) # NOTE
        render = np.reshape(rgb, hw + (3,))
        render = np.clip(render, 0, 1)
        render_out = join(FLAGS.out_dir, 'render.png')
        xm.io.img.write_arr(render, render_out)
        from IPython import embed; embed()


def get_config_ini(ckpt_path):
    return '/'.join(ckpt_path.split('/')[:-2]) + '.ini'


def make_datapipe(config, mode):
    dataset_name = config.get('DEFAULT', 'dataset')
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, mode, always_all_rays=True)
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)
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
