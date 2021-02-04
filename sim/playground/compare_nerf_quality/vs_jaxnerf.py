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
import json
import numpy as np
from tqdm import tqdm

from absl import app, flags

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string('my_result_root', '', "")
flags.DEFINE_string('jax_result_root', '', "")
flags.DEFINE_string('out_dir', '', "")
FLAGS = flags.FLAGS


def main(_):
    psnr_func = xm.metric.PSNR('float')
    psnr_mine, psnr_jax = [], []

    collage = []
    for batch_dir in tqdm(
            sorted(gfile.Glob(join(FLAGS.my_result_root, 'batch?????????'))),
            desc="Views"):
        metadata_json = join(batch_dir, 'metadata.json')
        with gfile.Open(metadata_json, 'rb') as h:
            metadata = json.load(h)
        mode, idx = metadata['id'].split('_')
        assert mode == 'val'

        # Ground truth
        gt_png = join(batch_dir, 'gt_rgb.png')
        gt = xm.io.img.load(gt_png)
        gt = xm.img.normalize_uint(gt)

        # My results
        pred_png = join(batch_dir, 'fine_rgb.png')
        pred_mine = xm.io.img.load(pred_png)
        pred_mine = xm.img.normalize_uint(pred_mine)
        psnr = psnr_func(gt, pred_mine)
        psnr_mine.append(psnr)
        pred_mine = xm.img.denormalize_float(pred_mine)
        pred_mine = xm.vis.text.put_text(pred_mine, '%.1f' % psnr)

        # JaxNeRF's results
        pred_png = join(FLAGS.jax_result_root, idx + '.png')
        pred_jax = xm.io.img.load(pred_png)
        pred_jax = xm.img.normalize_uint(pred_jax)
        psnr = psnr_func(gt, pred_jax)
        psnr_jax.append(psnr)
        pred_jax = xm.img.denormalize_float(pred_jax)
        pred_jax = xm.vis.text.put_text(pred_jax, '%.1f' % psnr)

        # Side by side
        gt = xm.img.denormalize_float(gt)
        collage_row = np.hstack((pred_jax, pred_mine, gt))
        collage.append(collage_row)

    collage = np.vstack(collage)
    collage_png = join(FLAGS.out_dir, 'mine-vs-jaxnerf.png')
    xm.io.img.write_img(collage, collage_png)

    # PSNR
    print("Mine average PSNR: %f" % np.average(psnr_mine))
    print("JaxNeRF average PSNR: %f" % np.average(psnr_jax))
    from IPython import embed; embed()


if __name__ == '__main__':
    app.run(main)
