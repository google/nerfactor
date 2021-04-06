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

# pylint: disable=cell-var-from-loop

from os.path import basename, dirname, join
from collections import OrderedDict
from itertools import product
from absl import app, flags

from google3.learning.deepmind.xmanager2.client import google as xm
from google3.learning.deepmind.xmanager import hyper
from google3.learning.deepmind.python.adhoc_import import binary_import
with binary_import.AutoGoogle3():
    from google3.learning.brain.frameworks.xmanager import xm_helper
    from google3.pyglib import gfile
    from google3.experimental.users.xiuming.sim.sim.util import \
        logging as logutil, config as configutil, io as ioutil


root = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim'

flags.DEFINE_string('user', 'xiuming', "Borg user")
flags.DEFINE_string(
    'outroot', join(root, 'output', 'train_s2021', '{config}_{dataset}_wo-geom-refine'), # NOTE
    "output root directory")
flags.DEFINE_integer('n_gpus', 8, "number of GPUs used per experiment")
flags.DEFINE_boolean('v100', True, "if false, use P100")
flags.DEFINE_string('cell', 'is', "")
FLAGS = flags.FLAGS

logger = logutil.Logger()


def main(_):
    mode = 'trainvali'
    # Base configurations
    base_config_ini = (
        'nerf_sfm_cam.ini',
        'nerf.ini',
        'brdf.ini', # BRDF pretraining
        'light.ini', # light pretraining
        'ns_shape.ini', # shape pretraining
        'ns_microfacet_gtlight.ini', # microfacet (minus lighting)
        'ns_gtlight.ini', # full (minus lighting)
        'ns_microfacet.ini', # microfacet
        'ns.ini', # full
        'ns_pixlight.ini', # full, but directly predicting light pixels
        'ns_microfacet_pixlight.ini', # microfacet, directly light pixels
    )[-2] # NOTE
    dataset = (
        'render/lego_interior_512',
        'render/hotdog_interior_512',
        'render_s2021/hotdog_2144', # bad geometry
        'render_s2021/hotdog_2188',
        'render_s2021/hotdog_2159', # bad geometry
        'render_s2021/hotdog_2234', # bad geometry
        'render_s2021/hotdog_3072',
        'render_s2021/hotdog_3083',
        'render_s2021/hotdog_2163',
        'render_s2021/hotdog_2171',
        'render_s2021/hotdog_probe_16-00_latlongmap',
        'render_s2021/lego_3072',
        'render_s2021/lego_3083',
        'render_s2021/lego_probe_16-00_latlongmap',
        'render_s2021/ficus_3072',
        'render_s2021/ficus_probe_16-00_latlongmap',
        'render_s2021/ficus_2188',
        'render_s2021/drums_3072',
        'render_s2021/drums_probe_16-00_latlongmap',
        'render_s2021/drums_2188',
        'real/hotdog_shady',
        'brdf/merl_npz/ims512_envmaph16_spp1',
        'brdf/merl_sep_npz/ims512_envmaph16_spp1',
        'envmaps/outdoor_npz_lh16',
    )[-5] # NOTE
    scene_id = dataset.split('/')[-1]
    dataset_no_slash = dataset.replace('/', '_')
    if base_config_ini in (
            'ns_microfacet_gtlight.ini', 'ns_gtlight.ini', 'ns_microfacet.ini',
            'ns.ini', 'ns_pixlight.ini', 'ns_microfacet_pixlight.ini'):
        epochs = 100
        ckpt_period = 10
        lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    elif base_config_ini in ('ns_shape.ini',):
        epochs = 200
        ckpt_period = 10
        lrs = [1e-3, 1e-2]
    elif base_config_ini in ('nerf.ini',):
        epochs = 2_000
        ckpt_period = 100
        lrs = [5e-4, 1e-3, 5e-3]
        if scene_id.startswith('ficus'):
            lrs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4] + lrs
    elif base_config_ini in ('brdf.ini',):
        epochs = 50_000
        ckpt_period = 100
        lrs = [1e-3, 1e-2]
    elif base_config_ini in ('light.ini',):
        epochs = 50_000
        ckpt_period = 100
        lrs = [1e-4, 1e-3]
    else:
        raise ValueError(base_config_ini)
    if scene_id.startswith(('hotdog_', 'lego_', 'drums_', 'ficus_')):
        near = 2
        far = 6
        if dataset.startswith('real/'):
            near = 0.1
            far = 2
    else:
        raise ValueError(scene_id)
    vali_period = ckpt_period
    vali_batches = 8
    imh = 512 if base_config_ini in ('nerf.ini', 'ns_shape.ini') else 256
    if base_config_ini == 'nerf.ini' and dataset == 'real/hotdog_shady':
        imh = 256 # since the images are wide
    # Shape
    data_nerf_root = join( # NOTE
        root, 'output', 'surf_s2021', '%s_lh16_spp1' % scene_id)
    shape_mode = (
        'nerf',
        'scratch',
        'finetune',
        'frozen',
    )[0] # NOTE
    shape_model_ckpt = join( # NOTE
        root, 'output', 'train_s2021',
        'ns_shape.ini_%s' % dataset_no_slash,
        'lr0.001' if dataset in (
            'render_s2021/lego_3072',
            'render_s2021/ficus_probe_16-00_latlongmap',
        ) else 'lr0.01',
        'checkpoints', 'ckpt-20')
    # BRDF or lighting
    albedo_slope = 0.7 # NOTE
    albedo_bias = 0.1 # NOTE
    brdf_model_ckpt = join( # NOTE
        root, 'output', 'train',
        'brdf.ini_brdf_merl_npz_ims512_envmaph16_spp1_new-vis', # raw
        # 'brdf.ini_brdf_merl_sep_npz_ims512_envmaph16_spp1', # separated
        'lr0.01', 'vis_test', 'ckpt-500')
    learned_brdf_scale = 1 # NOTE
    gt_light = join(
        root, 'data', 'envmaps', 'for-render_h16', 'train',
        '%s.hdr' % '_'.join(scene_id.split('_')[1:]))
    light_model_ckpt = join( # NOTE
        root, 'output', 'train', 'light.ini_envmaps_outdoor_npz_lh16',
        'lr0.001', 'vis_test', 'ckpt-50')
    light_scale = 1 # NOTE
    light_tv_weight = 5e-6 # NOTE
    light_achro_weight = 0 # NOTE
    # Smoothness, by spatial jittering
    nerf_shape_respect = 0.1 # NOTE
    if base_config_ini == 'ns_shape.ini':
        nerf_shape_respect = 1
    smooth_use_l1 = True
    smooth_weight = 0.1 # NOTE
    if base_config_ini == 'ns_shape.ini':
        smooth_weight = 0
    elif scene_id.startswith('lego_'):
        smooth_weight = 0.01 # NOTE
    #smooth_weight=1e-4#FIXME
    smooth_weight_brdf = (
        0,
        0.01, # better overall performance, but albedo might be dirty
        1e-6, # better albedo, since BRDF can be unsmooth to make albedo smooth
    )[1] # NOTE
    #smooth_weight_brdf=smooth_weight#FIXME
    xyz_jitter_std = 0.01
    # If value is a list, we will sweep that parameter
    override = OrderedDict({
        'imh': imh,
        'data_root': join(root, 'data', dataset),
        'white_bg': True,
        'near': near,
        'far': far,
        # Shape
        'data_nerf_root': data_nerf_root,
        'shape_mode': shape_mode,
        'shape_model_ckpt': shape_model_ckpt,
        'normal_loss_weight': nerf_shape_respect,
        'lvis_loss_weight': nerf_shape_respect,
        'normal_smooth_weight': smooth_weight,
        'lvis_smooth_weight': smooth_weight,
        # BRDF or lighting
        'pred_brdf': True,
        'albedo_slope': albedo_slope,
        'albedo_bias': albedo_bias,
        'albedo_smooth_weight': smooth_weight,
        'brdf_smooth_weight': smooth_weight_brdf,
        'brdf_model_ckpt': brdf_model_ckpt,
        'learned_brdf_scale': learned_brdf_scale,
        'rough_min': 0.1,
        'gt_light': gt_light,
        'light_model_ckpt': light_model_ckpt,
        'light_scale': light_scale,
        'light_tv_weight': light_tv_weight,
        'light_achro_weight': light_achro_weight,
        # Rendering
        'linear2srgb': True,
        # Spatial smoothness
        'xyz_jitter_std': xyz_jitter_std,
        'smooth_use_l1': smooth_use_l1,
        # Optimization
        'n_rays_per_step': 1024 * FLAGS.n_gpus,
        'outroot': FLAGS.outroot.format(
            config=base_config_ini, dataset=dataset_no_slash),
        'overwrite': False, # handled interactively down below
        'lr': lrs,
        'lr_decay_steps': 500_000,
        'clipnorm': -1,
        'clipvalue': -1,
        'epochs': epochs,
        'keep_recent_epochs': -1,
        'ckpt_period': ckpt_period,
        'vali_period': vali_period,
        'vali_batches': vali_batches})
    borg_params = {
        'user': FLAGS.user, 'n_gpus_per_exp': FLAGS.n_gpus, 'ram': 64 * xm.GiB,
        'cell': FLAGS.cell, 'priority': 200, 'n_cpus_per_exp': 10,
        'gpu_types': [xm.GpuType.V100] if FLAGS.v100 else [xm.GpuType.P100]}

    # Load configuration file as base parameters
    config_ini = join(dirname(__file__), 'config', base_config_ini)
    base_config = ioutil.read_config(config_ini)
    base_params = configutil.config2dict(base_config)

    # Find out parameters to sweep and use them to name the experiment folders
    sweep_k = []
    for k, v in override.items():
        if isinstance(v, list):
            sweep_k.append(k)
    override['xname'] = '_'.join('%s{%s}' % (x, x) for x in sweep_k)

    # Prepare output directory
    outroot = override.get('outroot', base_params['outroot'])
    cont = interact_prepare_dir(outroot, gfs_user=borg_params['user'])
    if not cont:
        # Shortcircuit if do not continue
        return
    exploration_name = basename(outroot)

    # Sweeping by generating multiple configuration files
    config_paths = replicate_config(config_ini, override, outroot)
    params = [hyper.sweep('config', hyper.categorical(config_paths))]
    params = hyper.product(params)

    # Requirements
    # additional_req = xm.BorgOverrides(disk=1 * xm.GiB)
    # additional_req.autopilot_params.fixed_ram = True
    req = xm.Requirements(
        cpu=borg_params['n_cpus_per_exp'], gpu=borg_params['n_gpus_per_exp'],
        gpu_types=borg_params['gpu_types'], ram=borg_params['ram'],
        # additional_requirements=additional_req,
        autopilot=True)

    # Runtime
    runtime = xm.Borg(
        cell=borg_params['cell'], priority=borg_params['priority'],
        borguser=borg_params['user'], requirements=req)
    runtime.overrides.xm_pass_arguments = True

    # Executable
    executable = xm.BuildTarget(
        '//experimental/users/xiuming/sim/sim:%s' % mode,
        platform=xm.Platform.GPU, runtime=runtime,
        args={'xprof_port': '%port_xprof%'})

    def work_unit_hyper_str(wid, param_dict):
        str_ = str(wid)
        str_ += '_%s' % param_dict['config']
        return str_

    # These experiments constitute an exploration
    exploration = xm_helper.parameter_sweep(
        executable, params,
        map_fns={'work_unit_hyper_str': work_unit_hyper_str})

    # Wrap exploration with TensorBoard (and MLDash)
    exploration = xm.WithTensorBoard(exploration, log_dir=outroot)
    exploration = xm_helper.WithMLDash(
        exploration, log_dir=outroot, experiment_name=exploration_name,
        gfs_user=borg_params['user'])

    # Launch exploration on Borg
    description = xm.ExperimentDescription(exploration_name)
    xm.launch_experiment(description, exploration)


def replicate_config(base_config_ini, override_dict, outdir):
    sweep_k, sweep_v = [], []
    for k, v in override_dict.items():
        if isinstance(v, list):
            sweep_k.append(k)
            sweep_v.append(v)

    # Each combination leads to a configuration file
    new_config_paths = []
    for comb in product(*sweep_v):
        config = ioutil.read_config(base_config_ini)

        # Override
        for k, v in override_dict.items():
            if k in sweep_k:
                # Use values in the combination, instead of a list of values
                v = comb[sweep_k.index(k)]
            config.set('DEFAULT', k, str(v))

        # Generate a new configuration file
        xname = override_dict['xname'].format(
            **{k: v for (k, v) in zip(sweep_k, comb)})
        new_config_ini = join(outdir, xname + '.ini')
        with gfile.Open(new_config_ini, 'w') as h:
            config.write(h)
        new_config_paths.append(new_config_ini)
        logger.info(
            "Replicated base configurations (with overriding) into:\n\t%s",
            new_config_ini)

    return new_config_paths


def interact_prepare_dir(dir_, gfs_user=None):
    if gfs_user is None:
        gfs_user = gfile.GetUser()

    if gfile.IsDirectory(dir_):
        # Directory already exists
        logger.info("Output directory exisits:\n\t%s", dir_)

        # Ask to confirm
        logger.warn((
            "Delete it and start from scratch (d), resume (r), or skip this "
            "(s)?"))
        need_input = True
        while need_input:
            response = input().lower()
            if response in ('d', 'r', 's'):
                need_input = False
            if need_input:
                logger.error("Enter only d, r, or s!")

        # Skip
        if response == 's':
            logger.info("Experiment skipped as requested")
            return False

        # Resume
        if response == 'r':
            logger.info("Experiment resumed from checkpoint")
            return True

        # Start from scratch
        try:
            with gfile.AsUser(gfs_user):
                gfile.DeleteRecursively(dir_)
        except gfile.GOSError:
            for u in [x for x in ('gcam-gpu', 'xiuming') if x != gfs_user]:
                try:
                    with gfile.AsUser(u):
                        gfile.DeleteRecursively(dir_)
                except gfile.GOSError:
                    pass
        logger.warn("Deleted and starting from scratch")

    with gfile.AsUser(gfs_user):
        gfile.MakeDirs(dir_)
    return True


if __name__ == '__main__':
    app.run(main)
