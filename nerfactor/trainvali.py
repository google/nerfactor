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

"""A general training and validation pipeline.
"""

from os.path import join, dirname, exists
from shutil import rmtree
from time import time
from collections import deque
from tqdm import tqdm
from absl import app, flags
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil


flags.DEFINE_string(
    'config', 'nerf.ini', "base .ini file in config/ or a full path")
flags.DEFINE_string('config_override', '', "e.g., 'key1=value1,key2=value2'")
flags.DEFINE_boolean('debug', False, "debug mode switch")
flags.DEFINE_enum(
    'device', 'gpu', ['cpu', 'gpu'], "running on what type of device(s)")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="trainvali")


def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    distributed_train_step_decor = distributed_train_step if FLAGS.debug \
        else tf.function(distributed_train_step)

    # Distribution strategy
    strategy = get_strategy()

    # Configurations
    config_ini = FLAGS.config
    if not exists(config_ini):
        config_ini = join(dirname(__file__), 'config', FLAGS.config)
    config = ioutil.read_config(config_ini)
    # Any override?
    if FLAGS.config_override != '':
        for kv in FLAGS.config_override.split(','):
            k, v = kv.split('=')
            config.set('DEFAULT', k, v)

    # Output directory
    config_dict = configutil.config2dict(config)
    xname = config.get('DEFAULT', 'xname').format(**config_dict)
    outroot = config.get('DEFAULT', 'outroot')
    outdir = join(outroot, xname)
    overwrite = config.getboolean('DEFAULT', 'overwrite')
    ioutil.prepare_outdir(outdir, overwrite=overwrite)
    logger.info("For results, see:\n\t%s", outdir)

    # Dump actual configuration used to disk
    config_out = outdir.rstrip('/') + '.ini'
    ioutil.write_config(config, config_out)

    # Make training dataset
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset_train = Dataset(config, 'train', debug=FLAGS.debug)
    global_bs_train = dataset_train.bs
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe_train = dataset_train.build_pipeline(no_batch=no_batch)
    datapipe_train = strategy.experimental_distribute_dataset(datapipe_train)

    # Make validation dataset
    dataset_vali = Dataset(config, 'vali', debug=FLAGS.debug)
    global_bs_vali = dataset_vali.bs # maybe different from training
    try:
        datapipe_vali = dataset_vali.build_pipeline(no_batch=no_batch)
    except FileNotFoundError:
        datapipe_vali = None
    # Sample validation batches, and just stick with them
    if datapipe_vali is None:
        vali_batches = None
    else:
        n_vali_batches = config.getint('DEFAULT', 'vali_batches')
        vali_batches = datapipe_vali.take(n_vali_batches)
        vali_batches = strategy.experimental_distribute_dataset(vali_batches)

    with strategy.scope():
        # Model
        model_name = config.get('DEFAULT', 'model')
        Model = models.get_model_class(model_name)
        model = Model(config, debug=FLAGS.debug)
        model.register_trainable()

        # Optimizer
        lr = config.getfloat('DEFAULT', 'lr')
        lr_decay_steps = config.getint('DEFAULT', 'lr_decay_steps', fallback=-1)
        if lr_decay_steps > 0:
            lr_decay_rate = config.getfloat('DEFAULT', 'lr_decay_rate')
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                lr, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate)
        kwargs = {'learning_rate': lr, 'amsgrad': True}
        clipnorm = config.getfloat('DEFAULT', 'clipnorm')
        clipvalue = config.getfloat('DEFAULT', 'clipvalue')
        err_msg = \
            "Both `clipnorm` and `clipvalue` are active -- turn one off"
        if clipnorm > 0:
            assert clipvalue < 0, err_msg
            kwargs['clipnorm'] = clipnorm
        if clipvalue > 0:
            assert clipnorm < 0, err_msg
            kwargs['clipvalue'] = clipvalue
        optimizer = tf.keras.optimizers.Adam(**kwargs)

        # Resume from checkpoint, if any
        ckptdir = join(outdir, 'checkpoints')
        assert model.trainable_registered, (
            "Register the trainable layers to have them tracked by the "
            "checkpoint")
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), optimizer=optimizer, net=model)
        keep_recent_epochs = config.getint('DEFAULT', 'keep_recent_epochs')
        if keep_recent_epochs <= 0:
            keep_recent_epochs = None # keep all epochs
        ckptmanager = tf.train.CheckpointManager(
            ckpt, ckptdir, max_to_keep=keep_recent_epochs)
        ckpt.restore(ckptmanager.latest_checkpoint)
        if ckptmanager.latest_checkpoint:
            logger.info(
                "Resumed from step:\n\t%s", ckptmanager.latest_checkpoint)
        else:
            logger.info("Started from scratch")

        # Summary directories
        writer_train = tf.summary.create_file_writer(
            join(outdir, 'summary_train'))
        writer_vali = tf.summary.create_file_writer(
            join(outdir, 'summary_vali'))
        train_vis_epoch_dir = join(outdir, 'vis_train', 'epoch{e:09d}')
        vali_vis_epoch_dir = join(outdir, 'vis_vali', 'epoch{e:09d}')
        train_vis_epoch_dir_deque = deque([], keep_recent_epochs)
        vali_vis_epoch_dir_deque = deque([], keep_recent_epochs)
        train_vis_batch_rawf = join(
            train_vis_epoch_dir, 'batch{b:09d}_raw.pickle')
        vali_vis_batch_rawf = join(
            vali_vis_epoch_dir, 'batch{b:09d}_raw.pickle')
        train_vis_batch_dir = join(train_vis_epoch_dir, 'batch{b:09d}')
        vali_vis_batch_dir = join(vali_vis_epoch_dir, 'batch{b:09d}')
        train_vis_batches_comp = join(train_vis_epoch_dir, 'all')
        vali_vis_batches_comp = join(vali_vis_epoch_dir, 'all') # add proper
        # extension yourself in your overriding function (this makes the
        # pipeline general and not specific to any model)

        # ====== Training loop ======
        epochs = config.getint('DEFAULT', 'epochs')
        vis_train_batches = config.getint('DEFAULT', 'vis_train_batches')
        ckpt_period = config.getint('DEFAULT', 'ckpt_period')
        vali_period = config.getint('DEFAULT', 'vali_period')
        step_restored = ckpt.step.numpy()
        for _ in tqdm(range(step_restored, epochs), desc="Training epochs"):

            # ------ Train on all batches ------
            batch_loss, batch_vis, batch_time = [], [], []
            for batch_i, batch in enumerate(datapipe_train):
                t0 = time()
                loss, to_vis = distributed_train_step_decor(
                    strategy, model, batch, optimizer, global_bs_train)
                batch_time.append(time() - t0)
                batch_loss.append(loss)
                if batch_i < vis_train_batches:
                    batch_vis.append(to_vis)
                if FLAGS.debug:
                    logger.warn(
                        "Debug mode: skipping the rest of this epoch")
                    break
            assert batch_time, "Dataset is empty"

            # Record step
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            # Checkpoint and summarize/visualize training
            if step % ckpt_period == 0:
                # Save checkpoint
                saved_path = ckptmanager.save()
                logger.info("Checkpointed step %s:\n\t%s", step, saved_path)
                # Summarize training
                with writer_train.as_default():
                    tf.summary.scalar(
                        "loss_train", tf.reduce_mean(batch_loss), step=step)
                    tf.summary.scalar(
                        "batch_time_train", tf.reduce_mean(batch_time),
                        step=step)
                    vis_dirs = []
                    for batch_i, to_vis in enumerate(batch_vis):
                        raw_f = train_vis_batch_rawf.format(e=step, b=batch_i)
                        vis_dir = train_vis_batch_dir.format(e=step, b=batch_i)
                        model.vis_batch(
                            to_vis, vis_dir, mode='train', dump_raw_to=raw_f)
                        vis_dirs.append(vis_dir)
                    # Generate a compilation (e.g., HTML) of all visualizations
                    comp_f = train_vis_batches_comp.format(e=step)
                    view_at = model.compile_batch_vis(
                        vis_dirs, comp_f, mode='train')
                    if view_at is not None:
                        tf.summary.text("vis_train", view_at, step=step)
                maintain_epoch_queue(
                    train_vis_epoch_dir_deque,
                    train_vis_epoch_dir.format(e=step))

            # ------ Validation ------
            if vali_batches is not None and vali_period > 0 \
                    and step % vali_period == 0:

                # Run validation on all validation batches
                batch_loss, batch_vis = [], []
                for batch_i, batch in enumerate(vali_batches):
                    # Validate on this validation batch
                    loss, to_vis = distributed_vali_step(
                        strategy, model, batch, global_bs_vali)
                    batch_loss.append(loss)
                    batch_vis.append(to_vis)

                # Summarize/visualize validation
                with writer_vali.as_default():
                    tf.summary.scalar(
                        "loss_vali", tf.reduce_mean(batch_loss), step=step)
                    vis_dirs = []
                    for batch_i, to_vis in enumerate(batch_vis):
                        raw_f = vali_vis_batch_rawf.format(e=step, b=batch_i)
                        vis_dir = vali_vis_batch_dir.format(e=step, b=batch_i)
                        model.vis_batch(
                            to_vis, vis_dir, mode='vali', dump_raw_to=raw_f)
                        vis_dirs.append(vis_dir)
                    # Generate a compilation (e.g., HTML) of all visualizations
                    comp_f = vali_vis_batches_comp.format(e=step)
                    view_at = model.compile_batch_vis(
                        vis_dirs, comp_f, mode='vali')
                    if view_at is not None:
                        tf.summary.text("vis_vali", view_at, step=step)
                maintain_epoch_queue(
                    vali_vis_epoch_dir_deque, vali_vis_epoch_dir.format(e=step))


def get_strategy():
    """Creates a distributed strategy.
    """
    strategy = None
    if FLAGS.device == 'cpu':
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
    elif FLAGS.device == 'gpu':
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise NotImplementedError(FLAGS.device)
    return strategy


# Will be decorated into a tf.function, depending on whether in debug mode
def distributed_train_step(strategy, model, batch, optimizer, global_bs):
    assert model.trainable_registered, \
        "Register the trainable layers before using `trainable_variables`"

    def train_step(batch):
        with tf.GradientTape() as tape:
            pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train')
            loss_kwargs['keep_batch'] = True # keep the batch dimension
            per_example_loss = model.compute_loss(pred, gt, **loss_kwargs)
            weighted_loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_bs)
        grads = tape.gradient(weighted_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return weighted_loss, partial_to_vis

    # Each GPU takes a step
    weighted_loss, partial_to_vis = strategy.run(train_step, args=(batch,))

    # Aggregate across GPUs
    loss, to_vis = aggeregate_dstributed(
        strategy, weighted_loss, partial_to_vis)

    return loss, to_vis


# Not using tf.function for validation step because it can become very slow
# when there is a long loop. Given validation step is likely called relatively
# infrequently, eager should be fine
def distributed_vali_step(strategy, model, batch, global_bs):
    def vali_step(batch):
        pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali')
        loss_kwargs['keep_batch'] = True # keep the batch dimension
        per_example_loss = model.compute_loss(pred, gt, **loss_kwargs)
        weighted_loss = tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=global_bs)
        return weighted_loss, partial_to_vis

    # Each GPU takes a step
    weighted_loss, partial_to_vis = strategy.run(vali_step, args=(batch,))

    # Aggregate across GPUs
    loss, to_vis = aggeregate_dstributed(
        strategy, weighted_loss, partial_to_vis)

    return loss, to_vis


def aggeregate_dstributed(strategy, weighted_loss, partial_to_vis):
    # Sum the weighted loss
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, weighted_loss, axis=None)

    # Concatenate the items to visualize back to the full batch
    to_vis = {}
    for k, v in partial_to_vis.items():
        to_vis[k] = tf.concat(
            tf.nest.flatten(v, expand_composites=True), axis=0)

    return loss, to_vis


def maintain_epoch_queue(queue, new_epoch_dir):
    queue.appendleft(new_epoch_dir)
    for epoch_dir in xm.os.sortglob(dirname(new_epoch_dir), '*'):
        if epoch_dir not in queue: # already evicted from queue (FIFO)
            rmtree(epoch_dir)


if __name__ == '__main__':
    app.run(main)
