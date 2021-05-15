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

from os.path import join, basename
from absl import app, flags
from tqdm import tqdm
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_boolean('color_correct_albedo', False, "")
flags.DEFINE_boolean('edit_albedo', False, "")
flags.DEFINE_boolean('edit_brdf', False, "")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="test")


def compute_rgb_scales(alpha_thres=0.9):
    """Computes RGB scales that match predicted albedo to ground truth,
    using just the first validation view.
    """
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # First validation view
    vali_dir = join(config_ini[:-4], 'vis_vali')
    data_root = config.get('DEFAULT', 'data_root')
    epoch_dirs = xm.os.sortglob(vali_dir, 'epoch?????????')
    epoch_dir = epoch_dirs[-1]
    batch_dirs = xm.os.sortglob(epoch_dir, 'batch?????????')
    batch_dir = batch_dirs[0]

    # Find GT path
    metadata_path = join(batch_dir, 'metadata.json')
    metadata = xm.io.json.load(metadata_path)
    view = metadata['id']
    pred_path = join(batch_dir, 'pred_albedo.png')
    gt_path = join(data_root, view, 'albedo.png')

    # Load prediction and GT
    pred = xm.io.img.read(pred_path) # gamma corrected
    gt = xm.io.img.read(gt_path) # linear
    pred = xm.img.normalize_uint(pred)
    gt = xm.img.normalize_uint(gt)
    pred = pred ** 2.2 # undo gamma
    gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')
    alpha = gt[:, :, 3]
    gt = gt[:, :, :3]

    # Compute color correction scales, in the linear space
    is_fg = alpha > alpha_thres
    opt_scale = []
    for i in range(3):
        x_hat = pred[:, :, i][is_fg]
        x = gt[:, :, i][is_fg]
        scale = x_hat.dot(x) / x_hat.dot(x_hat)
        opt_scale.append(scale)
    opt_scale = tf.convert_to_tensor(opt_scale, dtype=tf.float32)

    return opt_scale


def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_test', basename(FLAGS.ckpt))
    if FLAGS.edit_albedo:
        outroot = outroot.rstrip('/') + '_edit-albedo'
    if FLAGS.edit_brdf:
        outroot = outroot.rstrip('/') + '_edit-brdf'

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=FLAGS.debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    ioutil.restore_model(model, FLAGS.ckpt)

    # Optionally, color-correct the albedo
    albedo_scales = None
    if (not FLAGS.edit_albedo) and FLAGS.color_correct_albedo:
        albedo_scales = compute_rgb_scales()

    # Optionally, edit albedo and BRDF
    albedo_override = None
    if FLAGS.edit_albedo:
        # tgt_albedo = (0.913, 0.921, 0.925) # aluminium
        # tgt_albedo = (1, 0.843, 0) # gold
        tgt_albedo = (0, 1, 0) # green
        albedo_override = tf.convert_to_tensor(tgt_albedo, dtype=tf.float32)
    brdf_z_override = None
    if FLAGS.edit_brdf:
        # tgt_brdf = 'gold-metallic-paint1'
        tgt_brdf = 'pearl-paint'
        tgt_brdf_z = model.brdf_model.latent_code.z[
            model.brdf_model.brdf_names.index(tgt_brdf), :]
        brdf_z_override = tf.convert_to_tensor(tgt_brdf_z, dtype=tf.float32)

    # For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(
            tqdm(datapipe, desc="Inferring Views", total=n_views)):
        relight_olat = batch_i == n_views - 1 # only for the final view
        # Inference
        _, _, _, to_vis = model.call(
            batch, mode='test', relight_olat=relight_olat, relight_probes=True,
            albedo_scales=albedo_scales, albedo_override=albedo_override,
            brdf_z_override=brdf_z_override)
        # Visualize
        outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
        model.vis_batch(to_vis, outdir, mode='test', olat_vis=relight_olat)
        # Break if debugging
        if FLAGS.debug:
            break

    # Compile all visualized batches into a consolidated view (e.g., an
    # HTML or a video)
    batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
    outpref = outroot # proper extension should be added in the function below
    view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test')
    logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    app.run(main)
