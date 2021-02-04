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

# pylint: disable=unsupported-assignment-operation

from os.path import join, basename, dirname
import numpy as np
from tqdm import tqdm
from absl import app, flags

from google3.experimental.users.xiuming.sim.eval import util as eval_util
from google3.experimental.users.xiuming.sim.sim.util import light as light_util
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string('data_dir', '', "")
flags.DEFINE_string('run_dir', '', "")
flags.DEFINE_string('light_dir', '', "")
flags.DEFINE_string(
    'select_test_views', '000,025,050,075,100,125,150,175', "")
flags.DEFINE_string(
    'select_novel_illum',
    'olat-0000-0016,olat-0004-0000,olat-0004-0016,courtyard,sunrise', "")
flags.DEFINE_float('scale', 2, "")
flags.DEFINE_string('out_root', '', "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    scene = basename(FLAGS.data_dir.rstrip('/'))
    exp = basename(dirname(FLAGS.run_dir.rstrip('/')))

    outdir = join(FLAGS.out_root, exp)
    if xm.os.exists_isdir(outdir)[0]:
        xm.os.rm(outdir)

    # ------ Testing: figures

    select_test_views = None if FLAGS.select_test_views == '' \
        else [int(x) for x in FLAGS.select_test_views.split(',')]

    test_dir = join(FLAGS.run_dir, 'vis_test', 'ckpt-10')
    test_outdir = join(outdir, 'test')

    # For each view
    for batch_dir in tqdm(
            xm.os.sortglob(test_dir, 'batch?????????'),
            desc="Test views"):
        view_i = int(basename(batch_dir)[len('batch'):])
        if select_test_views is not None and view_i not in select_test_views:
            continue

        metadata_path = join(batch_dir, 'metadata.json')
        metadata = xm.io.json.load(metadata_path)
        view = metadata['id']
        gt_dir = join(FLAGS.data_dir, view)
        view = ''.join(view.split('_')) # remove underscores

        alpha_path = join(batch_dir, 'gt_alpha.png')
        alpha = eval_util.read_img(alpha_path)

        # For each light condition we have ground truth for
        for gt_path in xm.os.sortglob(gt_dir, 'rgba*'):
            gt_path_base = basename(gt_path)
            if gt_path_base == 'rgba.png':
                lname = 'orig'
                pred_path_base = 'pred_rgb.png'
            else:
                lname = gt_path_base.split('_')[1][:-len('.png')]
                pred_path_base = f'pred_rgb_{lname}.png'
            pred_path = join(batch_dir, pred_path_base)
            pred = eval_util.read_img(pred_path)
            gt = eval_util.read_img(gt_path, h=pred.shape[0])
            gt = xm.img.alpha_blend(gt, alpha, pred) # for pred.'s background
            matched_pred = eval_util.match_scale(pred, alpha, gt, per_ch=True)
            xm.io.img.write_arr(
                matched_pred, join(test_outdir, f'{view}_pred_{lname}.png'))
            xm.io.img.write_arr(
                gt, join(test_outdir, f'{view}_gt_{lname}.png'))

    # ------ Validation

    vali_dir = join(FLAGS.run_dir, 'vis_vali', 'epoch000000100')
    vali_outdir = join(outdir, 'vali')

    # Original illumination
    pred_path = join(vali_dir, 'pred_light.png')
    pred = xm.io.img.read(pred_path)
    pred = xm.img.normalize_uint(pred)
    vis_h = pred.shape[0]
    lname = scene.split('_')[1]
    if lname.endswith('_no-ambient'):
        lname = lname[:-len('_no-ambient')]
    gt_path = join(FLAGS.light_dir, 'train', lname + '.hdr')
    gt = xm.io.hdr.read(gt_path)
    gt = np.clip(gt, 0, np.inf)
    gt_uint = light_util.vis_light(gt, h=vis_h)
    xm.io.img.write_uint(gt_uint, join(vali_outdir, 'light_gt.png'))
    xm.io.img.write_float(pred, join(vali_outdir, 'light_pred.png'))

    # Novel illuminations
    for lname in FLAGS.select_novel_illum.split(','):
        if lname.startswith('olat-'):
            i, j = lname.split('-')[1:]
            i, j = int(i), int(j)
            light = np.zeros_like(gt)
            light[i, j, :] = 1
        else:
            light_path = join(FLAGS.light_dir, 'test', lname + '.hdr')
            light = xm.io.hdr.read(light_path)
        light = np.clip(light, 0, np.inf)
        light_uint = light_util.vis_light(light, h=vis_h)
        xm.io.img.write_uint(
            light_uint, join(vali_outdir, 'light_%s.png' % lname))

    # For each view
    for batch_dir in tqdm(
            xm.os.sortglob(vali_dir, 'batch?????????'),
            desc="Vali. views"):
        metadata_path = join(batch_dir, 'metadata.json')
        metadata = xm.io.json.load(metadata_path)
        view = metadata['id']
        gt_dir = join(FLAGS.data_dir, view)
        view = ''.join(view.split('_')) # remove underscores
        alpha_path = join(batch_dir, 'gt_alpha.png')
        alpha = eval_util.read_img(alpha_path)

        # Rendering
        pred_path = join(batch_dir, 'pred_rgb.png')
        gt_path = join(batch_dir, 'gt_rgb.png')
        pred = eval_util.read_img(pred_path)
        gt = eval_util.read_img(gt_path)
        xm.io.img.write_arr(
            pred, join(vali_outdir, '%s_render_pred.png' % view))
        xm.io.img.write_arr(
            gt, join(vali_outdir, '%s_render_gt.png' % view))

        # Albedo
        pred_path = join(batch_dir, 'pred_albedo.png')
        gt_path = join(gt_dir, 'albedo.png')
        pred = eval_util.read_img(pred_path)
        gt = eval_util.read_img(gt_path, h=pred.shape[0])
        gt = gt ** (1 / 2.2) # gamma correction
        gt = xm.img.alpha_blend(gt, alpha, pred) # for pred.'s background
        matched_pred = eval_util.match_scale(pred, alpha, gt, per_ch=True)
        xm.io.img.write_arr(
            matched_pred, join(vali_outdir, '%s_albedo_pred.png' % view))
        xm.io.img.write_arr(
            gt, join(vali_outdir, '%s_albedo_gt.png' % view))

        # Normals
        pred_path = join(batch_dir, 'pred_normal.png')
        gt_path = join(gt_dir, 'normal.png')
        pred = eval_util.read_img(pred_path)
        gt = eval_util.read_img(gt_path, h=pred.shape[0])
        gt = xm.img.alpha_blend(gt, alpha, pred) # for pred.'s background
        xm.io.img.write_arr(
            pred, join(vali_outdir, '%s_normal_pred.png' % view))
        xm.io.img.write_arr(
            gt, join(vali_outdir, '%s_normal_gt.png' % view))

        # Light visibility
        pred_path = join(batch_dir, 'pred_lvis.png')
        pred = eval_util.read_img(pred_path)
        xm.io.img.write_arr(
            pred, join(vali_outdir, '%s_lvis_pred.png' % view))

        # BRDF
        pred_path = join(batch_dir, 'pred_brdf.png')
        pred = eval_util.read_img(pred_path)
        xm.io.img.write_arr(
            pred, join(vali_outdir, '%s_brdf_pred.png' % view))


if __name__ == '__main__':
    app.run(main)
