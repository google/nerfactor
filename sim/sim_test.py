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

"""A Beam-on-Flume pipeline to run fully parallelized inference.
"""

from os.path import join, basename
from absl import app, flags
import apache_beam as beam

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.pipeline.flume.py import runner
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
from google3.experimental.users.xiuming.sim.sim import datasets, models
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil, config as configutil


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_enum('mode', 'local', ('local', 'launch', 'skip-inf'), "")
flags.DEFINE_boolean('color_correct_albedo', False, "")
flags.DEFINE_boolean('edit_albedo', False, "")
flags.DEFINE_boolean('edit_brdf', False, "")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="test")


def make_datapipe():
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=FLAGS.debug)

    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)
    return datapipe


def restore_model():
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)

    ioutil.restore_model(model, FLAGS.ckpt)

    return model


def get_batch_ind():
    logger.info(
        "Figuring out total number of batches by building a data pipeline")
    datapipe = make_datapipe()
    batch_ind = list(range(len(datapipe)))
    return batch_ind


def infer_batch(batch_i, total_batches=200):
    assert batch_i < total_batches, \
        f"Batch index ({batch_i}) out of bounds ({total_batches})"

    # Make dataset
    logger.info("Making the actual data pipeline")
    datapipe = make_datapipe()

    # Restore model
    logger.info("Restoring trained model")
    model = restore_model()

    # Output directory for this batch
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    outroot = join(config_ini[:-4], 'vis_test', basename(FLAGS.ckpt))
    if FLAGS.edit_albedo:
        outroot = outroot.rstrip('/') + '_edit-albedo'
    if FLAGS.edit_brdf:
        outroot = outroot.rstrip('/') + '_edit-brdf'
    outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))

    # Optionally color-correct the albedo
    albedo_scales = None
    if (not FLAGS.edit_albedo) and FLAGS.color_correct_albedo:
        albedo_scales = compute_rgb_scales()

    # Optionally edit albedo and BRDF
    src_albedo = (1.000, 0.766, 0.336) # gold
    tgt_albedo = (0.913, 0.921, 0.925) # aluminium
    src_brdf_z = (-0.8937415, -1.4565116, -0.8024932) # gold
    tgt_brdf_z = (1.6404238, -0.172426, -0.5986401) # aluminium
    src_albedo = tf.convert_to_tensor(src_albedo, dtype=tf.float32)
    tgt_albedo = tf.convert_to_tensor(tgt_albedo, dtype=tf.float32)
    src_brdf_z = tf.convert_to_tensor(src_brdf_z, dtype=tf.float32)
    tgt_brdf_z = tf.convert_to_tensor(tgt_brdf_z, dtype=tf.float32)
    a = float(batch_i) / total_batches
    albedo_override = None
    if FLAGS.edit_albedo:
        albedo_override = a * tgt_albedo + (1 - a) * src_albedo
    brdf_z_override = None
    if FLAGS.edit_brdf:
        brdf_z_override = a * tgt_brdf_z + (1 - a) * src_brdf_z

    # Run inference on a single batch
    logger.info("Running inference")
    for batch in datapipe.skip(batch_i).take(1):
        _, _, _, to_vis = model.call(
            batch, mode='test', albedo_scales=albedo_scales,
            albedo_override=albedo_override, brdf_z_override=brdf_z_override)

    # Visualize
    logger.info("Running visualization")
    outdir = outdir.format(i=batch_i)
    model.vis_batch(to_vis, outdir, mode='test')


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


def pipeline_local():
    batch_ind = get_batch_ind()
    for bi in batch_ind:
        infer_batch(bi)
        logger.warn(
            "Returning after only one batch, because this is local testing")
        return


def pipeline(root):
    """Beam-on-Flume pipeline.
    """
    batch_ind = get_batch_ind()
    batch_ind_pcol = root | 'create_input' >> beam.Create(batch_ind)
    _ = batch_ind_pcol | beam.ParDo(infer_batch)


def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Parallel inference on all batches
    if FLAGS.mode == 'local':
        pipeline_local()
    elif FLAGS.mode == 'launch':
        runner.FlumeRunner().run(pipeline)

    # Compile all visualized batches into a consolidated view (e.g., an
    # HTML or a video)
    model = restore_model()
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    outroot = join(config_ini[:-4], 'vis_test', basename(FLAGS.ckpt))
    if FLAGS.edit_albedo:
        outroot = outroot.rstrip('/') + '_edit-albedo'
    if FLAGS.edit_brdf:
        outroot = outroot.rstrip('/') + '_edit-brdf'
    batch_vis_dirs = join(outroot, 'batch?????????')
    batch_vis_dirs = sorted(gfile.Glob(batch_vis_dirs))
    outpref = outroot # proper extension should be added in the function below
    view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode='test')
    logger.info("Compilation available for viewing at\n\t%s", view_at)


if __name__ == '__main__':
    app.run(main)
