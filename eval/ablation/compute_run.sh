#!/usr/bin/env bash
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


set -e

cd "$(p4 g4d)"

blaze run -c opt \
    'experimental/users/xiuming/sim/eval/ablation:compute' \
    -- \
    --data_root='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/hotdog_3072' \
    --data_root='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/data/render_s2021/hotdog_3072_no-ambient' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072/lr0.005/' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8/lr0.005/' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv1e-6/lr0.005/' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_probe_16-00_latlongmap_tv-5e-6/lr0.001/' \
    --ours_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap/lr0.001' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072/lr0.005/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8/lr0.005/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv/lr0.0005/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.005/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6/lr0.001/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_hotdog_probe_16-00_latlongmap_bsmooth1e-6/lr0.005/' \
    --wo_learned_brdf_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_microfacet_pixlight.ini_render_s2021_ficus_probe_16-00_latlongmap_bsmooth1e-6/lr0.0005' \
    --wo_geom_refine_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_wo-geom-refine/lr0.005/' \
    --wo_geom_refine_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8_wo-geom-refine/lr0.005/' \
    --wo_geom_refine_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv_wo-geom-refine/lr0.001/' \
    --wo_geom_refine_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-refine/lr0.0005/' \
    --wo_geom_refine_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-refine/lr0.001/' \
    --wo_geom_pretrain_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_wo-geom-pretrain/lr0.005/' \
    --wo_geom_pretrain_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8_wo-geom-pretrain/lr0.001/' \
    --wo_geom_pretrain_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv_wo-geom-pretrain/lr0.005/' \
    --wo_geom_pretrain_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-pretrain/lr0.005/' \
    --wo_geom_pretrain_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-geom-pretrain/lr0.001/' \
    --wo_smooth_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_wo-smooth/lr0.005/' \
    --wo_smooth_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_albedo-0.1-0.8_wo-smooth/lr0.005/' \
    --wo_smooth_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv_wo-smooth/lr0.005/' \
    --wo_smooth_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-smooth/lr0.005/' \
    --wo_smooth_dir='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/train_s2021/ns_pixlight.ini_render_s2021_hotdog_3072_tv-5e-6_wo-smooth/lr0.001/' \
    --out_root='/cns/is-d/home/gcam-eng/gcam/interns/xiuming/sim/output/results_s2021/ablation' \
    --albedo_per_ch_scale \
    --relight_scale \
    --gfs_user='gcam-gpu' \
    "$@"
    #--relight_per_ch_scale \

cd -
