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
from tqdm import tqdm

from absl import app, flags

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

from google3.experimental.users.xiuming.sim.eval import util


flags.DEFINE_string('data_root', '', "")
flags.DEFINE_float('alpha_thres', 0.9, "")
flags.DEFINE_string('est1_dir', '', "")
flags.DEFINE_string('est2_dir', '', "")
flags.DEFINE_string('est3_dir', '', "")
flags.DEFINE_string('out_root', '', "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    metrics = ('psnr', 'ssim', 'lpips') # NOTE: same order as paper table

    if xm.os.exists_isdir(FLAGS.out_root)[0]:
        xm.os.rm(FLAGS.out_root)

    calculators = {
        'ssim': xm.metric.SSIM(float),
        'psnr': xm.metric.PSNR(float),
        'lpips': xm.metric.LPIPS(float)}

    imgs = {'est1': [], 'est2': [], 'est3': [], 'gt': []}

    for est in tqdm(('est1', 'est2', 'est3'), desc="Estimations"):
        run_dir = getattr(FLAGS, est + '_dir')
        # Get the latest epoch
        vali_epoch_dirs = xm.os.sortglob(
            join(run_dir, 'vis_vali'), 'epoch?????????')
        vali_epoch_dir = vali_epoch_dirs[-1]

        # ------ For each validation view
        for batch_dir in tqdm(
                xm.os.sortglob(vali_epoch_dir, 'batch*'), desc="Vali. views"):
            metadata_path = join(batch_dir, 'metadata.json')
            pred_albedo_path = join(batch_dir, 'pred_albedo.png')
            metadata = xm.io.json.load(metadata_path)
            gt_albedo_path = join(FLAGS.data_root, metadata['id'], 'albedo.png')
            # Load
            pred_gamma = util.read_img(pred_albedo_path) # gamma corrected
            pred_linear = pred_gamma ** 2.2 # linear
            gt_linear = util.read_img(
                gt_albedo_path, force_white_bg=True,
                h=pred_linear.shape[0]) # linear
            # Scale predicted albedo
            gt_alpha_path = join(batch_dir, 'gt_alpha.png')
            alpha = util.read_img(gt_alpha_path)
            scaled_pred_linear = util.match_scale(
                pred_linear, alpha, gt_linear, alpha_thres=FLAGS.alpha_thres,
                per_ch=True)
            scaled_pred_gamma = \
                scaled_pred_linear ** (1 / 2.2) # gamma corrected now
            gt_gamma = gt_linear ** (1 / 2.2) # gamma corrected now
            # Save what are used in score computation, for qualitative fig.
            out_dir = join(FLAGS.out_root, metadata['id'])
            xm.io.img.write_float(
                scaled_pred_gamma, join(out_dir, f'{est}_pred_albedo.png'))
            if est == 'est1': # do this just once
                xm.io.img.write_float(
                    gt_gamma, join(out_dir, f'{est}_gt_albedo.png'))
            #
            imgs[est].append(scaled_pred_gamma)
            if est == 'est1': # do this just once
                imgs['gt'].append(gt_gamma)

    # Confusion matrix
    for metric in tqdm(metrics, desc="Metrics"):
        calc = calculators[metric]
        sum_mat, cnt_mat = np.zeros((4, 4)), np.zeros((4, 4))
        for img_tuple in zip(imgs['est1'], imgs['est2'], imgs['est3'], imgs['gt']):
            for i in range(sum_mat.shape[0]):
                for j in range(sum_mat.shape[1]):
                    score = calc(img_tuple[i], img_tuple[j])
                    sum_mat[i, j] += min(score, 100)
                    cnt_mat[i, j] += 1
        avg_mat = sum_mat / cnt_mat
        out_png = f'/usr/local/home/xiuming/Desktop/consistency_{metric}.png'
        plot_confusion_mat(avg_mat, out_png)
        print(metric, avg_mat)

    from IPython import embed; embed()


def plot_confusion_mat(mat, outpath, h_pix=512):
    h = mat.shape[0]
    pix_per_cell = int(h_pix / h)
    heatmap = []
    for i in range(mat.shape[0]):
        heatmap_row = []
        for j in range(mat.shape[1]):
            block = mat[i, j] * np.ones((pix_per_cell, pix_per_cell))
            heatmap_row.append(block)
        heatmap_row = np.hstack(heatmap_row)
        heatmap.append(heatmap_row)
    heatmap = np.vstack(heatmap)
    xm.vis.matrix.matrix_as_heatmap(heatmap, outpath=outpath)


if __name__ == '__main__':
    app.run(main)
