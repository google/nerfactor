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


flags.DEFINE_string('pred_dir', '', "")
flags.DEFINE_string('gt_root', '', "")
flags.DEFINE_string('light_root', '', "")
flags.DEFINE_string('out_root', '', "")
flags.DEFINE_float('alpha_thres', 0.9, "")
flags.DEFINE_float('light_gamma', 4, "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    select = (
        'orig', 'olat-0000-0024', 'olat-0004-0000', 'olat-0004-0016',
        'courtyard', 'interior')

    if xm.os.exists_isdir(FLAGS.out_root)[0]:
        xm.os.rm(FLAGS.out_root)

    scene = basename(FLAGS.gt_root.rstrip('/'))
    outdir = join(FLAGS.out_root, scene)

    metadata_path = join(FLAGS.pred_dir, 'metadata.json')
    metadata = xm.io.json.load(metadata_path)
    view = metadata['id']

    light_h = None

    for lname in select:
        # Prediction
        if lname == 'orig':
            pred_path = join(FLAGS.pred_dir, 'pred_rgb.png')
        else:
            pred_path = join(FLAGS.pred_dir, 'pred_rgb_%s.png' % lname)
        pred = util.read_img(pred_path)

        # Ground truth
        if lname == 'orig':
            gt_path = join(FLAGS.gt_root, view, 'rgba.png')
        else:
            gt_path = join(FLAGS.gt_root, view, 'rgba_%s.png' % lname)
        gt = util.read_img(gt_path, h=pred.shape[0], keep_alpha=True)
        alpha = gt[:, :, 3]
        gt = gt[:, :, :3]

        # Scale and color tone matching
        pred_scaled = util.match_scale(
            pred, alpha, gt, alpha_thres=FLAGS.alpha_thres, per_ch=False)
        pred_scaled_ch = util.match_scale(
            pred, alpha, gt, alpha_thres=FLAGS.alpha_thres, per_ch=True)

        # Light
        if lname == 'orig':
            orig_lname = scene.split('_')[1]
            if orig_lname.endswith('_no-ambient'):
                orig_lname = orig_lname[:-len('_no-ambient')]
            light_path = join(FLAGS.light_root, 'train', orig_lname + '.hdr')
            light = xm.io.hdr.read(light_path)
            light_h = light.shape[0]
        else:
            if lname.startswith('olat-'):
                i, j = lname.split('-')[1:]
                i, j = int(i), int(j)
                assert light_h is not None, \
                    "Load some light first before getting here"
                light = np.zeros((light_h, 2 * light_h, 3), dtype=float)
                light[i, j, :] = 1
            else:
                light_path = join(FLAGS.light_root, 'test', lname + '.hdr')
                light = xm.io.hdr.read(light_path)
                light_h = light.shape[0]
        light = np.clip(light, 0, np.inf)
        light = xm.img.resize(light, new_w=gt.shape[1], method='tf')
        light = (light / light.max()) ** (1 / FLAGS.light_gamma)

        # Write to disk
        xm.io.img.write_float(light, join(outdir, '%s.png' % lname))
        xm.io.img.write_float(gt, join(outdir, '%s_gt.png' % lname))
        xm.io.img.write_float(pred, join(outdir, '%s_pred.png' % lname))
        xm.io.img.write_float(
            pred_scaled, join(outdir, '%s_pred_scaled.png' % lname))
        xm.io.img.write_float(
            pred_scaled_ch, join(outdir, '%s_pred_scaled-per-ch.png' % lname))


if __name__ == '__main__':
    app.run(main)
