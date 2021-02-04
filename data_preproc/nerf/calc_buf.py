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
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from absl import app, flags
import apache_beam as beam

from google3.pyglib import gfile
from google3.pipeline.flume.py import runner
from google3.experimental.users.xiuming.sim.brdf.renderer import gen_light_xyz
from google3.experimental.users.xiuming.sim.sim import datasets, models
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil, config as configutil, img as imgutil
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_string('data_root', '', "input data root")
flags.DEFINE_integer('spp', 1, "samples per pixel")
flags.DEFINE_integer(
    'light_h', 16, "number of pixels along environment map's height (latitude)")
flags.DEFINE_string('out_root', '', "output root")
flags.DEFINE_enum('mode', 'local', ('local', 'flume'), "")
flags.DEFINE_integer('fps', 12, "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="calc_buf")


def gen_args():
    args = []
    if FLAGS.debug:
        args.append(('train', 33))
        return args
    for mode in ('train', 'vali', 'test'):
        if mode == 'vali':
            mode_str = 'val'
        else:
            mode_str = mode
        for config_dir in sorted(gfile.Glob(
                join(FLAGS.data_root, f'{mode_str}_???'))):
            i = int(basename(config_dir).split('_')[1])
            args.append((mode, i))
    assert args, "No arguments generated"
    return args


def pipeline_local():
    args = gen_args()
    for arg_tuple in args:
        job(arg_tuple)
        logger.warn("Returning after one point, because this is local testing")
        return


def pipeline(root):
    """Beam-on-Flume pipeline.
    """
    args = gen_args()
    args_pcol = root | 'create_input' >> beam.Create(args)
    _ = args_pcol | beam.ParDo(job)


def job(arg_tuple):
    mode, skip = arg_tuple

    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Make dataset
    datapipe = make_datapipe(config, mode)
    sps = int(np.sqrt(FLAGS.spp)) # no need to check if square

    # Restore model
    model = restore_model(config, FLAGS.ckpt)

    # Run inference on a single batch
    for batch in datapipe.skip(skip).take(1):
        id_, hw, rayo, rayd, _ = batch
        hw = hw[0, :]

        rayd = tf.linalg.l2_normalize(rayd, axis=1)

        out_dir = join(FLAGS.out_root, id_[0].numpy().decode())
        if not gfile.Exists(out_dir):
            gfile.MakeDirs(out_dir)
        logger.info("Output directory:\n\t%s", out_dir)

        # ------ Camera to object

        logger.info("Alpha, XYZ, & Normal")

        if FLAGS.debug:
            with gfile.Open(
                    '/usr/local/home/xiuming/Desktop/occu.npy',
                    'rb') as h:
                occu = np.load(h)
            occu = tf.convert_to_tensor(occu)
            with gfile.Open(
                    '/usr/local/home/xiuming/Desktop/exp_depth.npy',
                    'rb') as h:
                exp_depth = np.load(h)
            exp_depth = tf.convert_to_tensor(exp_depth)
            with gfile.Open(
                    '/usr/local/home/xiuming/Desktop/exp_normal.npy',
                    'rb') as h:
                exp_normal = np.load(h)
            exp_normal = tf.convert_to_tensor(exp_normal)
        else:
            occu, exp_depth, exp_normal = compute_depth_and_normal(
                model, rayo, rayd, config)

        # Write alpha map
        alpha_map = tf.reshape(occu, hw * sps)
        alpha_map = average_supersamples(alpha_map, sps)
        alpha_map = tf.clip_by_value(alpha_map, 0., 1.)
        write_alpha(alpha_map, out_dir)

        # Write XYZ map, whose background filling value is (0, 0, 0)
        surf = rayo + rayd * exp_depth[:, None] # Surface XYZs
        xyz_map = tf.reshape(surf, (hw[0] * sps, hw[1] * sps, 3))
        xyz_map = average_supersamples(xyz_map, sps)
        xyz_map = imgutil.alpha_blend(xyz_map, alpha_map)
        write_xyz(xyz_map, out_dir)

        # Write normal map, whose background filling value is (0, 1, 0),
        # since using (0, 0, 0) leads to (0, 0, 0) tangents
        normal_map = tf.reshape(exp_normal, (hw[0] * sps, hw[1] * sps, 3))
        normal_map = average_supersamples(normal_map, sps)
        normal_map_bg = tf.convert_to_tensor((0, 1, 0), dtype=tf.float32)
        normal_map_bg = tf.tile(normal_map_bg[None, None, :], tuple(hw) + (1,))
        normal_map = imgutil.alpha_blend(normal_map, alpha_map, normal_map_bg)
        normal_map = tf.linalg.l2_normalize(normal_map, axis=2)
        write_normal(normal_map, out_dir)

        # ------ Object to light

        logger.info("Light Visibility")

        # Don't waste memory on those "miss" rays
        hit = tf.reshape(alpha_map, (-1,)) > 0.
        surf = tf.boolean_mask(surf, hit, axis=0)
        normal = tf.boolean_mask(exp_normal, hit, axis=0)

        lvis_hit = compute_light_visibility( # (n_surf_pts, n_lights)
            model, surf, normal, rayo, config, lpix_chunk=64)
        n_lights = lvis_hit.shape[1]

        # Put the light visibility values into the full tensor
        hit_map = hit.numpy().reshape(tuple(hw) + (1,))
        lvis = np.zeros( # (imh, imw, n_lights)
            tuple(hw) + (n_lights,), dtype=np.float32)
        lvis[np.broadcast_to(hit_map, lvis.shape)] = lvis_hit.ravel()

        # Write light visibility map
        write_lvis(lvis, out_dir)


def average_supersamples(map_supersampled, sps):
    maps = []
    for i in range(sps):
        for j in range(sps):
            sample = map_supersampled[i::sps, j::sps, ...]
            sample = sample[None, ...]
            maps.append(sample)
    maps = tf.concat(maps, axis=0)
    return tf.reduce_mean(maps, axis=0)


def compute_light_visibility(
        model, surf, normal, rayo, config, lvis_near=.1, lpix_chunk=128):
    mlp_chunk = config.getint('DEFAULT', 'mlp_chunk')
    n_samples_coarse = config.getint('DEFAULT', 'n_samples_coarse')
    n_samples_fine = config.getint('DEFAULT', 'n_samples_fine')
    lin_in_disp = config.getboolean('DEFAULT', 'lin_in_disp')
    perturb = config.getboolean('DEFAULT', 'perturb')
    near = config.getfloat('DEFAULT', 'near')

    light_w = 2 * FLAGS.light_h
    lxyz, lareas = gen_light_xyz(FLAGS.light_h, light_w)
    lxyz = tf.convert_to_tensor(lxyz.astype(np.float32))
    lareas = tf.convert_to_tensor(lareas.astype(np.float32))
    lxyz_flat = tf.reshape(lxyz, (1, -1, 3))

    n_lights = lxyz_flat.shape[1]
    lvis_hit = np.zeros(
        (surf.shape[0], n_lights), dtype=np.float32) # (n_surf_pts, n_lights)
    for i in tqdm(range(0, n_lights, lpix_chunk), desc="Light pixel chunks"):
        end_i = min(n_lights, i + lpix_chunk)
        lxyz_chunk = lxyz_flat[:, i:end_i, :] # (1, lpix_chunk, 3)

        # From surface to lights
        surf2l = lxyz_chunk - surf[:, None, :] # (n_surf_pts, lpix_chunk, 3)
        surf2l = tf.math.l2_normalize(surf2l, axis=2)
        surf2l_flat = tf.reshape(surf2l, (-1, 3)) # (n_surf_pts * lpix_chunk, 3)

        surf_rep = tf.tile(surf[:, None, :], (1, surf2l.shape[1], 1))
        surf_flat = tf.reshape(surf_rep, (-1, 3)) # (n_surf_pts * lpix_chunk, 3)

        # Save memory by ignoring back-lit points
        lcos = tf.einsum('ijk,ik->ij', surf2l, normal)
        front_lit = lcos > 0 # (n_surf_pts, lpix_chunk)
        front_lit_flat = tf.reshape(
            front_lit, (-1,)) # (n_surf_pts * lpix_chunk)
        surf_flat_frontlit = tf.boolean_mask(surf_flat, front_lit_flat, axis=0)
        surf2l_flat_frontlit = tf.boolean_mask( # (n_frontlit_pairs, 3)
            surf2l_flat, front_lit_flat, axis=0)

        # Query coarse model
        cam_dist = tf.reduce_mean(tf.linalg.norm(rayo, axis=1))
        z = model.gen_z( # NOTE: start from lvis_near instead of 0
            lvis_near, cam_dist - near, n_samples_coarse,
            surf2l_flat_frontlit.shape[0], lin_in_disp=lin_in_disp,
            perturb=perturb)
        pts = surf_flat_frontlit[:, None, :] + \
            surf2l_flat_frontlit[:, None, :] * z[:, :, None]
        pts_flat = tf.reshape(pts, (-1, 3))
        sigma_flat = eval_sigma_mlp(
            model, pts_flat, mlp_chunk=mlp_chunk, use_fine=False)
        sigma = tf.reshape(sigma_flat, pts.shape[:2])
        weights = model.accumulate_sigma(sigma, z, surf2l_flat_frontlit)

        # Obtain additional samples using importance sampling
        z = model.gen_z_fine(z, weights, n_samples_fine, perturb=perturb)
        pts = surf_flat_frontlit[:, None, :] + \
            surf2l_flat_frontlit[:, None, :] * z[:, :, None]
        pts_flat = tf.reshape(pts, (-1, 3))

        # Evaluate all samples with the fine model
        sigma_flat = eval_sigma_mlp(
            model, pts_flat, mlp_chunk=mlp_chunk, use_fine=True)
        sigma = tf.reshape(sigma_flat, pts.shape[:2])
        weights = model.accumulate_sigma(sigma, z, surf2l_flat_frontlit)
        occu = tf.reduce_sum(weights, -1) # (n_frontlit_pairs,)

        # Put the light visibility values into the full tensor
        front_lit_full = np.zeros(lvis_hit.shape, dtype=bool)
        front_lit_full[:, i:end_i] = front_lit.numpy()
        lvis_hit[front_lit_full] = 1 - occu.numpy()

    return lvis_hit # (n_surf_pts, n_lights)


def compute_depth_and_normal(model, rayo, rayd, config):
    mlp_chunk = config.getint('DEFAULT', 'mlp_chunk')
    n_samples_coarse = config.getint('DEFAULT', 'n_samples_coarse')
    n_samples_fine = config.getint('DEFAULT', 'n_samples_fine')
    lin_in_disp = config.getboolean('DEFAULT', 'lin_in_disp')
    perturb = config.getboolean('DEFAULT', 'perturb')
    near = config.getfloat('DEFAULT', 'near')
    far = config.getfloat('DEFAULT', 'far')

    # Points in space to evaluate the coarse model at
    z = model.gen_z(
        near, far, n_samples_coarse, rayo.shape[0], lin_in_disp=lin_in_disp,
        perturb=perturb)
    pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None] # shape is
    # (n_rays, n_samples, 3)
    pts_flat = tf.reshape(pts, (-1, 3))

    # Evaluate coarse model for importance sampling
    sigma_flat = eval_sigma_mlp(
        model, pts_flat, mlp_chunk=mlp_chunk, use_fine=False)
    sigma = tf.reshape(sigma_flat, pts.shape[:2])
    weights = model.accumulate_sigma(sigma, z, rayd)

    # Obtain additional samples using importance sampling
    z = model.gen_z_fine(z, weights, n_samples_fine, perturb=perturb)
    pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None]
    pts_flat = tf.reshape(pts, (-1, 3))

    # Evaluate all samples with the fine model
    embedder = model.embedder['xyz']
    fine_enc = model.net['fine_enc']
    fine_sigma_out = model.net.get(
        'fine_a_out', model.net['fine_sigma_out'])
    sigma_chunks, normal_chunks = [], [] # chunk by chunk to avoid OOM
    for i in tqdm(range(0, pts_flat.shape[0], mlp_chunk), desc="Fine chunks"):
        end_i = min(pts_flat.shape[0], i + mlp_chunk)
        pts_chunk = pts_flat[i:end_i, :]
        # Sigma
        with tf.GradientTape() as g:
            g.watch(pts_chunk)
            sigma_chunk = tf.nn.relu(
                fine_sigma_out(fine_enc(embedder(pts_chunk))))
        # Normals: derivatives of sigma
        bjac_chunk = g.batch_jacobian(sigma_chunk, pts_chunk)
        bjac_chunk = tf.reshape(
            bjac_chunk, (bjac_chunk.shape[0], 3)) # safe squeezing
        normal_chunk = -tf.linalg.l2_normalize(bjac_chunk, axis=1)
        #
        sigma_chunks.append(sigma_chunk)
        normal_chunks.append(normal_chunk)
    sigma_flat = tf.concat(sigma_chunks, axis=0)
    normal_flat = tf.concat(normal_chunks, axis=0)
    sigma = tf.reshape(sigma_flat, pts.shape[:2]) # (n_rays, n_samples)
    normal = tf.reshape(normal_flat, pts.shape) # (n_rays, n_samples, 3)

    # Accumulate samples into expected depth and normals
    weights = model.accumulate_sigma(sigma, z, rayd) # (n_rays, n_samples)
    occu = tf.reduce_sum(weights, -1) # (n_rays,)
    # Estimated depth is expected distance
    exp_depth = tf.reduce_sum(weights * z, axis=-1) # (n_rays,)
    # Computed weighted normal along each ray
    exp_normal = tf.reduce_sum(weights[:, :, None] * normal, axis=-2)

    return occu, exp_depth, exp_normal


def eval_sigma_mlp(model, pts, mlp_chunk=65536, use_fine=False):
    embedder = model.embedder['xyz']
    if use_fine:
        pref = 'fine_'
        desc = "Fine chunks"
    else:
        pref = 'coarse_'
        desc = "Coarse chunks"
    enc = model.net[pref + 'enc']
    sigma_out = model.net.get(pref + 'a_out', model.net[pref + 'sigma_out'])

    # Chunk by chunk to avoid OOM
    sigma_chunks = []
    for i in tqdm(range(0, pts.shape[0], mlp_chunk), desc=desc):
        end_i = min(pts.shape[0], i + mlp_chunk)
        pts_chunk = pts[i:end_i, :]
        sigma_chunk = tf.nn.relu(sigma_out(enc(embedder(pts_chunk))))
        sigma_chunks.append(sigma_chunk)
    sigma = tf.concat(sigma_chunks, axis=0)

    return sigma


def main(_):
    # Parallel inference on all batches
    if FLAGS.mode == 'local':
        pipeline_local()
    else:
        runner.FlumeRunner().run(pipeline)


def write_lvis(lvis, out_dir):
    # Dump raw
    raw_out = join(out_dir, 'lvis.npy')
    with gfile.Open(raw_out, 'wb') as h:
        np.save(h, lvis)

    # Visualize the average across all lights as an image
    vis_out = join(out_dir, 'lvis.png')
    lvis_avg = np.mean(lvis, axis=2)
    xm.io.img.write_arr(lvis_avg, vis_out, clip=True)

    # Visualize light visibility for each light pixel
    vis_out = join(out_dir, 'lvis.webm')
    frames = []
    for i in range(lvis.shape[2]): # for each light pixel
        frame = np.clip(lvis[:, :, i], 0, 1)
        frame = xm.img.denormalize_float(frame)
        frame = np.dstack([frame] * 3)
        frames.append(frame)
    ioutil.write_video(frames, vis_out, fps=FLAGS.fps)


def write_xyz(xyz_arr, out_dir):
    arr = xyz_arr.numpy()

    # Dump raw
    raw_out = join(out_dir, 'xyz.npy')
    with gfile.Open(raw_out, 'wb') as h:
        np.save(h, arr)

    # Visualization
    vis_out = join(out_dir, 'xyz.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    xm.io.img.write_arr(arr_norm, vis_out)


def write_normal(arr, out_dir):
    arr = arr.numpy()

    # Dump raw
    raw_out = join(out_dir, 'normal.npy')
    with gfile.Open(raw_out, 'wb') as h:
        np.save(h, arr)

    # Visualization
    vis_out = join(out_dir, 'normal.png')
    arr = (arr + 1) / 2
    xm.io.img.write_arr(arr, vis_out, clip=True)


def write_alpha(arr, out_dir):
    arr = arr.numpy()
    vis_out = join(out_dir, 'alpha.png')
    xm.io.img.write_arr(arr, vis_out)


def make_datapipe(config, mode):
    dataset_name = config.get('DEFAULT', 'dataset')
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, mode, always_all_rays=True, spp=FLAGS.spp)
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)
    return datapipe


def restore_model(config, ckpt_path):
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config)

    ioutil.restore_model(model, ckpt_path)

    return model


if __name__ == '__main__':
    app.run(main)
