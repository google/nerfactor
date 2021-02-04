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

from absl import app, flags

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

from google3.experimental.users.xiuming.sim.eval import util


flags.DEFINE_string('data_root', '', "")
flags.DEFINE_string('sirfs_root', '', "")
flags.DEFINE_string('nerf_env_root', '', "")
flags.DEFINE_string('out_root', '', "")
flags.DEFINE_float('alpha_thres', 0.9, "")
flags.DEFINE_boolean('albedo_per_ch_scale', True, "")
flags.DEFINE_string(
    'select_test_views', '000,025,050,075,100,125,150,175', "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def sirfs():
    buffers = ('normal', 'albedo')
    normal_metrics = ('angle',) # NOTE: same order as paper table
    metrics = ('psnr', 'ssim', 'lpips') # NOTE: same order as paper table
    fmt = '%.4f'

    out_root = join(FLAGS.out_root, 'sirfs')
    if xm.os.exists_isdir(out_root)[0]:
        xm.os.rm(out_root)

    calculators = {
        'ssim': xm.metric.SSIM(float),
        'psnr': xm.metric.PSNR(float),
        'lpips': xm.metric.LPIPS(float)}

    # results: buffer->metric
    results = {}
    for buffer_ in buffers:
        results[buffer_] = {}
        buffer_metrics = normal_metrics if buffer_ == 'normal' else metrics
        for metric in buffer_metrics:
            results[buffer_][metric] = []

    for scene_dir in tqdm(xm.os.sortglob(FLAGS.sirfs_root, '*'), desc="Scenes"):
        scene = basename(scene_dir)
        for pred_normal_path in xm.os.sortglob(
                scene_dir, 'val_???_normal.png'):
            view = basename(pred_normal_path)[:len('val_???')]
            pred_albedo_path = join(scene_dir, '%s_reflectance.png' % view)
            gt_rgba_path = join(FLAGS.data_root, scene, view, 'rgba.png')
            gt_albedo_path = join(FLAGS.data_root, scene, view, 'albedo.png')
            gt_normal_path = join(FLAGS.data_root, scene, view, 'normal.png')

            # Load
            alpha = util.read_img(gt_rgba_path, keep_alpha=True)[:, :, -1]
            pred, gt = {}, {}
            pred['albedo'] = util.read_img(pred_albedo_path)
            pred['normal'] = util.read_img(pred_normal_path)
            gt['normal'] = util.read_img(gt_normal_path, force_white_bg=True)
            gt['albedo'] = util.read_img(
                gt_albedo_path, force_white_bg=True) # NOTE: raw values; not tonemapped
            pred['albedo'] = xm.img.alpha_blend(
                pred['albedo'], alpha, np.ones_like(pred['albedo']))
            pred['normal'] = xm.img.alpha_blend(
                pred['normal'], alpha, np.ones_like(pred['normal']))

            # Scale predicted albedo
            pred['albedo'] = util.match_scale(
                pred['albedo'], alpha, gt['albedo'],
                alpha_thres=FLAGS.alpha_thres, per_ch=FLAGS.albedo_per_ch_scale)
            gt['albedo'] = gt['albedo'] ** (1 / 2.2) # gamma correction
            pred['albedo'] = pred['albedo'] ** (1 / 2.2) # gamma correction

            # Save what are used in score computation, for qualitative fig.
            out_dir = join(out_root, scene)
            xm.io.img.write_float(
                pred['albedo'], join(out_dir, f'{view}_pred_albedo.png'))
            xm.io.img.write_float(
                gt['albedo'], join(out_dir, f'{view}_gt_albedo.png'))
            xm.io.img.write_float(
                pred['normal'], join(out_dir, f'{view}_pred_normal.png'))
            xm.io.img.write_float(
                gt['normal'], join(out_dir, f'{view}_gt_normal.png'))

            # Compute scores
            for buffer_ in [x for x in buffers if not x.startswith('relight_')]:
                buffer_metrics = normal_metrics if buffer_ == 'normal' \
                    else metrics
                for metric in buffer_metrics:
                    if metric == 'angle':
                        error = util.avg_angle(
                            pred[buffer_], gt[buffer_], alpha)
                    else:
                        calc = calculators[metric]
                        error = calc(pred[buffer_], gt[buffer_])
                    results[buffer_][metric].append(error)

    for buffer_, metric2scores in results.items():
        print("%%%%%%%%%%%%%%%%", buffer_)
        for metric, scores in metric2scores.items():
            print(metric, fmt % np.mean(scores))


def nerf_env():
    # NOTE: orders are important
    models = ('ours',)
    tasks = ('relight_pt', 'relight_img')
    metrics = ('psnr', 'ssim', 'lpips')
    fmt = '%.4f'
    sorted_lnames = (
        'city',
        'courtyard',
        'forest',
        'interior',
        'night',
        'olat-0000-0000',
        'olat-0000-0008',
        'olat-0000-0016',
        'olat-0000-0024',
        'olat-0004-0000',
        'olat-0004-0008',
        'olat-0004-0016',
        'olat-0004-0024',
        'studio',
        'sunrise',
        'sunset',
    )

    out_root = join(FLAGS.out_root, 'nerf_env')
    if xm.os.exists_isdir(out_root)[0]:
        xm.os.rm(out_root)

    calculators = {
        'ssim': xm.metric.SSIM(float),
        'psnr': xm.metric.PSNR(float),
        'lpips': xm.metric.LPIPS(float)}

    # results: metric->task->scores
    results = {}
    for metric in metrics:
        results[metric] = {}
        for task in tasks:
            results[metric][task] = []

    for scene_dir in tqdm(xm.os.sortglob(FLAGS.nerf_env_root, '*'), desc="Scenes"):
        scene = basename(scene_dir)
        if scene == 'hotdog_2159':
            continue

        for illum_dir in xm.os.sortglob(join(scene_dir, scene), '??'):
            lid = basename(illum_dir)
            lname = sorted_lnames[int(lid)]

            for test_view in FLAGS.select_test_views.split(','):
                pred_path = join(illum_dir, 'pred_%s.png' % test_view)
                gt_path = join(
                    FLAGS.data_root, scene, 'test_' + test_view,
                    'rgba_%s.png' % lname)
                pred = util.read_img(pred_path)
                gt = util.read_img(
                    gt_path, h=pred.shape[0], keep_alpha=True)
                alpha = gt[:, :, 3]
                gt = gt[:, :, :3]
                gt = xm.img.alpha_blend(gt, alpha, pred) # composite

                # Compute scores
                for metric in metrics:
                    calc = calculators[metric]
                    error = calc(gt, pred)
                    if lname.startswith('olat-'):
                        results[metric]['relight_pt'].append(error)
                    else:
                        results[metric]['relight_img'].append(error)

    from IPython import embed; embed()
    for metric, task2scores in results.items():
        print("%%%%%%%%%%%%%%%%", metric)
        for task, scores in task2scores.items():
            print(task, fmt % np.mean(scores))


def main(_):
    nerf_env()
    return

    sirfs()


if __name__ == '__main__':
    app.run(main)
