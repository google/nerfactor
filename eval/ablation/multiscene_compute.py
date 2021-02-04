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


flags.DEFINE_string('out_root', '', "")
flags.DEFINE_float('alpha_thres', 0.9, "")
flags.DEFINE_boolean('albedo_per_ch_scale', False, "")
flags.DEFINE_boolean('relight_scale', False, "")
flags.DEFINE_boolean('relight_per_ch_scale', False, "")
flags.DEFINE_integer('n_test_views', 8, "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    data_roots = (
        '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/hotdog_3072_no-ambient',
        '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/ficus_probe_16-00_latlongmap',
        '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/drums_2188',
    )
    dirs = {
        'ours': (
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap/lr0.001/',
            #'/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188/lr0.001/',
        ),
        'wo_learned_brdf': (
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_bsmooth1e-6/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_drums_2188_bsmooth1e-6/lr0.001/',
        ),
        'wo_geom_refine': (
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-refine/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-geom-refine/lr0.001/',
            #'/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-geom-refine/lr0.001/',
        ),
        'wo_geom_pretrain': (
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-pretrain/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-geom-pretrain/lr0.001/',
            #'/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-geom-pretrain/lr0.001/',
        ),
        'wo_smooth': (
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-smooth/lr0.001/',
            '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-smooth/lr0.001/',
            #'/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-smooth/lr0.001/',
        )}

    models = (
        'wo_learned_brdf', 'ours', 'wo_geom_refine', 'wo_geom_pretrain',
        'wo_smooth',)
    buffers = ( # NOTE: same order as paper table
        'normal', 'albedo', 'recon', 'relight_pt', 'relight_img')
    normal_metrics = ('angle',) # NOTE: same order as paper table
    metrics = ('psnr', 'ssim', 'lpips') # NOTE: same order as paper table
    top_k = min(3, len(models))
    fmt = '%.4f'

    if xm.os.exists_isdir(FLAGS.out_root)[0]:
        xm.os.rm(FLAGS.out_root)

    calculators = {
        'ssim': xm.metric.SSIM(float),
        'psnr': xm.metric.PSNR(float),
        'lpips': xm.metric.LPIPS(float)}

    # results: model->buffer->metric
    results = {}
    for model in models:
        results[model] = {}
        for buffer_ in buffers:
            results[model][buffer_] = {}
            buffer_metrics = normal_metrics if buffer_ == 'normal' else metrics
            for metric in buffer_metrics:
                results[model][buffer_][metric] = []

    for model in tqdm(models, desc="Models"):
        run_dirs = dirs[model]
        if len(run_dirs) == 2:
            data_roots_ = data_roots[:2]
        else:
            data_roots_ = data_roots

        for data_root, run_dir in zip(data_roots_, run_dirs):
            # Get the latest epoch
            vali_epoch_dirs = xm.os.sortglob(
                join(run_dir, 'vis_vali'), 'epoch?????????')
            vali_epoch_dir = vali_epoch_dirs[-1]
            test_ckpt_dirs = xm.os.sortglob(join(run_dir, 'vis_test'), 'ckpt-??')
            if not test_ckpt_dirs:
                test_ckpt_dirs = xm.os.sortglob(join(run_dir, 'vis_test'), 'ckpt-?')
            assert test_ckpt_dirs, "No test results: %s" % run_dir
            test_ckpt_dir = test_ckpt_dirs[-1]

            # ------ For each test view (relighting)
            batch_dirs = xm.os.sortglob(test_ckpt_dir, 'batch*')
            batch_dirs = batch_dirs[::(len(batch_dirs) // FLAGS.n_test_views)]
            for batch_dir in tqdm(batch_dirs, desc="Test views"):
                metadata_path = join(batch_dir, 'metadata.json')
                metadata = xm.io.json.load(metadata_path)
                data_view_dir = join(data_root, metadata['id'])
                alpha_path = join(batch_dir, 'gt_alpha.png') # NeRF's
                alpha = util.read_img(alpha_path)
                # Load relit predictions and GT
                pred, gt = {}, {}
                for gt_path in xm.os.sortglob(data_view_dir, 'rgba_*.png'):
                    lname = basename(gt_path)[len('rgba_'):-len('.png')]
                    pred_path = join(batch_dir, f'pred_rgb_{lname}.png')
                    pred_srgb = util.read_img(pred_path)
                    gt_srgb = util.read_img(gt_path, h=pred_srgb.shape[0])
                    gt_srgb = xm.img.alpha_blend( # NOTE: composite GT on pred.
                        gt_srgb, alpha, pred_srgb) # for its colored background
                    pred_linear = xm.img.srgb2linear(pred_srgb)
                    gt_linear = xm.img.srgb2linear(gt_srgb)
                    # Scale matching in the linear space
                    if FLAGS.relight_scale: # NOTE
                        scaled_pred_linear = util.match_scale(
                            pred_linear, alpha, gt_linear,
                            per_ch=FLAGS.relight_per_ch_scale)
                    else:
                        scaled_pred_linear = pred_linear
                    pred[lname] = xm.img.linear2srgb(scaled_pred_linear) # sRGB now
                    gt[lname] = xm.img.linear2srgb(gt_linear)
                # Save what are used in score computation, for qualitative fig.
                out_dir = join(FLAGS.out_root, metadata['id'])
                for k, v in gt.items():
                    xm.io.img.write_float(
                        pred[k], join(out_dir, f'{model}_pred_rgb_{k}.png'))
                    xm.io.img.write_float(
                        v, join(out_dir, f'{model}_gt_rgb_{k}.png'))
                # Compute scores
                for lname, pred_ in pred.items():
                    gt_ = gt[lname]
                    for metric in metrics:
                        calc = calculators[metric]
                        error = calc(pred_, gt_)
                        if lname.startswith('olat-'):
                            results[model]['relight_pt'][metric].append(error)
                        else:
                            results[model]['relight_img'][metric].append(error)

            # ------ For each validation view (albedo, normals, view synthesis)
            for batch_dir in tqdm(
                    xm.os.sortglob(vali_epoch_dir, 'batch*'), desc="Vali. views"):
                metadata_path = join(batch_dir, 'metadata.json')
                pred_albedo_path = join(batch_dir, 'pred_albedo.png')
                pred_normal_path = join(batch_dir, 'pred_normal.png')
                pred_recon_path = join(batch_dir, 'pred_rgb.png')
                gt_recon_path = join(batch_dir, 'gt_rgb.png')
                metadata = xm.io.json.load(metadata_path)
                gt_albedo_path = join(data_root, metadata['id'], 'albedo.png')
                gt_normal_path = join(data_root, metadata['id'], 'normal.png')
                # Load
                pred, gt = {}, {}
                pred_albedo = util.read_img(pred_albedo_path) # gamma corrected
                pred['albedo'] = pred_albedo ** 2.2 # linear
                pred['normal'] = util.read_img(pred_normal_path)
                pred['recon'] = util.read_img(pred_recon_path) # sRGB
                gt['recon'] = util.read_img(gt_recon_path)
                gt['normal'] = util.read_img(
                    gt_normal_path, force_white_bg=True,
                    h=pred['normal'].shape[0])
                gt['albedo'] = util.read_img(
                    gt_albedo_path, force_white_bg=True,
                    h=pred['albedo'].shape[0]) # linear
                # Scale predicted albedo
                gt_alpha_path = join(batch_dir, 'gt_alpha.png')
                alpha = util.read_img(gt_alpha_path)
                scaled_pred_linear = util.match_scale(
                    pred['albedo'], alpha, gt['albedo'],
                    alpha_thres=FLAGS.alpha_thres, per_ch=FLAGS.albedo_per_ch_scale)
                pred['albedo'] = \
                    scaled_pred_linear ** (1 / 2.2) # gamma corrected now
                gt['albedo'] = gt['albedo'] ** (1 / 2.2) # gamma corrected now
                # Save what are used in score computation, for qualitative fig.
                out_dir = join(FLAGS.out_root, metadata['id'])
                xm.io.img.write_float(
                    pred['albedo'], join(out_dir, f'{model}_pred_albedo.png'))
                xm.io.img.write_float(
                    gt['albedo'], join(out_dir, f'{model}_gt_albedo.png'))
                xm.io.img.write_float(
                    pred['normal'], join(out_dir, f'{model}_pred_normal.png'))
                xm.io.img.write_float(
                    gt['normal'], join(out_dir, f'{model}_gt_normal.png'))
                xm.io.img.write_float(
                    pred['recon'], join(out_dir, f'{model}_pred_recon.png'))
                xm.io.img.write_float(
                    gt['recon'], join(out_dir, f'{model}_gt_recon.png'))
                # Compute scores
                for buffer_ in [x for x in buffers if not x.startswith('relight_')]:
                    buffer_metrics = normal_metrics if buffer_ == 'normal' \
                        else metrics
                    for metric in buffer_metrics:
                        if metric == 'angle':
                            error = util.avg_angle(pred[buffer_], gt[buffer_], alpha)
                        else:
                            calc = calculators[metric]
                            error = calc(pred[buffer_], gt[buffer_])
                        results[model][buffer_][metric].append(error)

    # Compute averages and get top 3 for each column (i.e., buffer-metric)
    cols, heats = [], []
    for buffer_ in buffers:
        buffer_metrics = normal_metrics if buffer_ == 'normal' else metrics
        for metric in buffer_metrics:
            col = []
            higher_is_better = metric in ('psnr', 'ssim')
            for model in models:
                numbers = results[model][buffer_][metric]
                assert numbers, \
                    f"Error list is empty for {metric}->{model}->{buffer_}"
                avg = np.mean(numbers)
                col.append(avg)
            col = np.array(col).reshape(-1, 1)
            cols.append(col)
            # Compute heat level
            heat = np.zeros_like(col)
            ind, _ = xm.sig.get_extrema(col, top=higher_is_better, n=top_k)
            for i in range(top_k):
                heat[ind[0][i], ind[1][i]] = top_k - i
            heats.append(heat)
    cols = np.hstack(cols)
    heats = np.hstack(heats)

    # Generate LaTeX rows that can be directly copied and pasted
    print("\n%%%%%%%%%%%%%%%%")
    for row_i, model in enumerate(models):
        if model == 'ours':
            row = '\model '
        elif model == 'wo_learned_brdf':
            row = '\model (microfacet) '
        elif model == 'wo_geom_refine':
            row = 'w/o geom. refine. '
        elif model == 'wo_geom_pretrain':
            row = 'w/o geom. pretrain. '
        elif model == 'wo_smooth':
            row = 'w/o smoothness '
        else:
            raise ValueError(model)
        for avg, heat in zip(cols[row_i, :], heats[row_i, :]):
            cell_str = util.get_colored_cell_str(heat)
            cell_str = cell_str % fmt
            row += cell_str % avg
        row += ' \\\\'
        print(row)
    print("%%%%%%%%%%%%%%%%\n")

    from IPython import embed; embed()


if __name__ == '__main__':
    app.run(main)
