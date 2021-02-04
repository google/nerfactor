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

"""Simple script that figures out the latest checkpoint and then replicates
it to the test result folder (so that the checkpoint won't get deleted by the
training manager while the Flume test jobs are still running).

This is standalone and not part of `test.py` because that script may be
relaunched by Borg repeatedly, leading to inconsistent "latest" checkpoints.
"""

from os.path import join, basename
from absl import app, flags

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from google3.pyglib import gfile
from google3.experimental.users.xiuming.sim.sim.util import io as ioutil, \
    logging as logutil


flags.DEFINE_string('id', '', "")
flags.DEFINE_string(
    'out_file', '/path/to/out_file',
    "path to the output file containing the latest checkpoint's replica")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="grab_latest_ckpt")


def get_best(id_):
    root = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim'
    outdir = join(root, 'output')

    # NeRF
    if id_ in ('nerf_hotdog_2159',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_2159',
            'lr0.0005.ini')
    elif id_ in ('nerf_hotdog_2234',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_2234',
            'lr0.0005.ini')
    elif id_ in ('nerf_hotdog_3072',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_3072',
            'lr0.0005.ini')
    elif id_ in ('nerf_hotdog_3083',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_3083',
            'lr0.001.ini')
    elif id_ in ('nerf_hotdog_probe_16-00_latlongmap',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_probe_16-00_latlongmap',
            'lr0.0005.ini')
    elif id_ in ('nerf_hotdog_2188',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_2188_run2',
            'lr0.0005.ini')
    elif id_ in ('nerf_hotdog_2163',):
        ini = join(
            outdir, 'train_s2021',
            'nerf.ini_render_s2021_hotdog_2163_run3/lr0.005.ini')
    elif id_ == 'nerf_lego_3083':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_lego_3083/lr0.001.ini'
    elif id_ == 'nerf_lego_3072':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_lego_3072/lr0.001.ini'
    elif id_ == 'nerf_lego_probe_16-00_latlongmap':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_lego_probe_16-00_latlongmap/lr0.001.ini'
    elif id_ == 'nerf_drums_2188':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_drums_2188/lr0.0005.ini'
    elif id_ == 'nerf_drums_probe_16-00_latlongmap':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_drums_probe_16-00_latlongmap/lr0.001.ini'
    elif id_ == 'nerf_ficus_2188':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_ficus_2188/lr0.0005.ini'
    elif id_ == 'nerf_ficus_probe_16-00_latlongmap':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_ficus_probe_16-00_latlongmap/lr0.0001.ini'
    elif id_ == 'nerf_ficus_3072':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/nerf.ini_render_s2021_ficus_3072/lr0.0001.ini'

    # BRDF pretraining
    elif id_ in ('brdf',):
        ini = join(
            outdir, 'train',
            'brdf.ini_brdf_merl_npz_ims512_envmaph16_spp1_new-vis', # raw
            # 'brdf.ini_brdf_merl_sep_npz_ims512_envmaph16_spp1', # separated
            'lr0.01.ini')

    # Light pretraining
    elif id_ in ('light',):
        ini = join(
            outdir, 'train',
            'light.ini_envmaps_outdoor_npz_lh16',
            'lr0.001.ini')

    # Microfacet (GT light)
    elif id_ in ('microfacet_gtlight_hotdog_interior',):
        ini = join(
            outdir, 'train',
            'ns_microfacet_gtlight.ini_render_hotdog_interior_512_gtlight-not-trainable',
            # 'ns_microfacet_gtlight.ini_render_hotdog_interior_512',
            'lr0.005.ini')

    # Microfacet
    elif id_ in ('microfacet_hotdog_interior',):
        ini = join(
            outdir, 'train',
            'ns_microfacet.ini_render_hotdog_interior_512',
            'lr0.0001.ini')
    elif id_ in ('microfacet_hotdog_probe_16-00_latlongmap',):
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_probe_16-00_latlongmap_bsmooth1e-6/lr0.005.ini'
    elif id_ == 'microfacet_hotdog_3072':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001.ini'
    elif id_ == 'microfacet_hotdog_2188':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_2188_bsmooth1e-6/lr0.005.ini'
    elif id_ == 'microfacet_hotdog_2163':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_2163_bsmooth1e-6/lr0.0005.ini'
    elif id_ == 'microfacet_ficus_probe_16-00_latlongmap':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_bsmooth1e-6/lr0.0005.ini'

    # NS (GT light)
    elif id_ in ('ns_gtlight_hotdog_interior',):
        ini = join(
            outdir, 'train',
            'ns_gtlight.ini_render_hotdog_interior_512',
            'lr0.001.ini')
    elif id_ in ('ns_gtlight_hotdog_interior_sep-brdf',):
        ini = join(
            outdir, 'train',
            'ns_gtlight.ini_render_hotdog_interior_512_sep-brdf',
            'lr0.005.ini')

    # NS
    elif id_ in ('ns_hotdog_interior',):
        ini = join(
            outdir, 'train',
            'ns.ini_render_hotdog_interior_512',
            'lr0.001.ini')
    elif id_ in ('ns_hotdog_interior_sep-brdf',):
        ini = join(
            outdir, 'train',
            'ns.ini_render_hotdog_interior_512_sep-brdf',
            'lr0.005.ini')
    elif id_ in ('ns_hotdog_2159',):
        ini = join(
            outdir, 'train_s2021',
            'ns.ini_render_s2021_hotdog_2159',
            'lr0.001.ini')

    # Ours
    elif id_ in ('ours_hotdog_3072',):
        ini = join(
            outdir, 'train_s2021',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_brdf-scale-1_brdf-smooth-1e-6',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_brdf-scale-1_brdf-smooth-0.01',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8_brdf-smooth-1e-6',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8/lr0.005.ini',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_tv1e-6',
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_tv8e-6/lr0.001.ini'
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_tv2e-6/lr0.005.ini'
            # 'ns_pixlight.ini_render_s2021_hotdog_3072_tv4e-6/lr0.005.ini'
            'ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001.ini')
    elif id_ in ('ours_wo-learned-brdf',):
        ini = join(
            outdir, 'train_s2021',
            'ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001.ini')
    elif id_ in ('ours_wo-geom-refine',):
        ini = join(
            outdir, 'train_s2021',
            'ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-refine/lr0.001.ini')
    elif id_ in ('ours_wo-geom-pretrain',):
        ini = join(
            outdir, 'train_s2021',
            'ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-pretrain/lr0.001.ini')
    elif id_ in ('ours_wo-smooth',):
        ini = join(
            outdir, 'train_s2021',
            'ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-smooth/lr0.001.ini')
    elif id_ == 'ours_lego_wo-learned-brdf':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_lego_3072/lr0.005.ini'
    elif id_ == 'ours_lego_wo-smooth':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_lego_3072_wo-smooth/lr0.001.ini'
    elif id_ == 'ours_lego_wo-geom-pretrain':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_lego_3072_wo-geom-pretrain/lr0.005.ini'
    elif id_ == 'ours_lego_wo-geom-refine':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_lego_3072_wo-geom-refine/lr0.005.ini'

    elif id_ == 'ours_ficus_wo-learned-brdf':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_bsmooth1e-6/lr0.001.ini'
    elif id_ == 'ours_ficus_wo-smooth':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-smooth/lr0.001.ini'
    elif id_ == 'ours_ficus_wo-geom-pretrain':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-geom-pretrain/lr0.001.ini'
    elif id_ == 'ours_ficus_wo-geom-refine':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_wo-geom-refine/lr0.001.ini'

    elif id_ == 'ours_drums_wo-learned-brdf':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_drums_2188_bsmooth1e-6/lr0.001.ini'
    elif id_ == 'ours_drums_wo-smooth':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-smooth/lr0.001.ini'
    elif id_ == 'ours_drums_wo-geom-pretrain':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-geom-pretrain/lr0.001.ini'
    elif id_ == 'ours_drums_wo-geom-refine':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188_wo-geom-refine/lr0.001.ini'

    elif id_ in ('ours_hotdog_2188',):
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_2188_tv-5e-6/lr0.001.ini'
    elif id_ in ('ours_hotdog_3083',):
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3083_tv-5e-6/lr0.001.ini'
    elif id_ in ('ours_hotdog_probe_16-00_latlongmap',):
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_probe_16-00_latlongmap_tv-5e-6/lr0.001.ini'
    elif id_ in ('ours_lego_3072',):
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_lego_3072_smooth0.01/lr0.001.ini'
    elif id_ == 'ours_hotdog_probe_16-00_latlongmap':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_probe_16-00_latlongmap_tv-5e-6/lr0.001.ini'
    elif id_ == 'ours_hotdog_2163':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_2163_tv-5e-6/lr0.005.ini'
    elif id_ == 'ours_ficus_probe_16-00_latlongmap':
        # ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap/lr0.001.ini'
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap/lr0.005.ini'
    elif id_ == 'ours_drums_2188':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_drums_2188/lr0.001.ini'
    elif id_ == 'ours_ficus_3072':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_3072/lr0.001.ini'
    elif id_ == 'ours_ficus_2188':
        ini = '/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_2188/lr0.001.ini'

    else:
        raise ValueError(id_)

    return ini


def main(_):
    config_ini = get_best(FLAGS.id)
    xdir = config_ini[:-len('.ini')]

    # Get latest
    ckpt_dir = join(xdir, 'checkpoints')
    assert gfile.Exists(ckpt_dir), \
        "Checkpoint directory doesn't exist:\n\t%s" % ckpt_dir
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

    # Replicate it in case it gets evicted as the model trains
    cp_to = join(xdir, 'vis_test')
    ioutil.prepare_outdir(cp_to, overwrite=False, quiet=True)
    for f in gfile.Glob(latest_ckpt + '*'):
        gfile.Copy(f, join(cp_to, basename(f)), overwrite=True)
    logger.info(
        "Replicated latest checkpoint\n\t%s\nto\n\t%s", latest_ckpt, cp_to)

    # Persist the path to disk for the Flume test jobs to read from
    replica_path = join(cp_to, basename(latest_ckpt))
    with gfile.Open(FLAGS.out_file, 'wb') as h:
        h.write(replica_path)
    logger.info((
        "File containing path to the (replicated) latest checkpoint persisted "
        "to\n\t%s\nfor Flume workers to read"), FLAGS.out_file)


if __name__ == '__main__':
    app.run(main)
