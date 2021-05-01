#!/usr/bin/env python

import sys
from os import makedirs
from os.path import join, exists, basename
from glob import glob
from absl import app, flags
import xiuminglib as xm


flags.DEFINE_string(
        'scene_file',
        '/data/vision/billf/mooncam/output/xiuming/sun-earth-moon/scene/lat42.3601_lng-71.0589',
        "")
flags.DEFINE_string(
        'earth_img_dir',
        '/data/vision/billf/mooncam/output/xiuming/reflectance/earth-images/',
        "")
flags.DEFINE_integer('days', 4, "")
flags.DEFINE_integer('spp', 64, "")
flags.DEFINE_integer('imh', 256, "")
flags.DEFINE_string(
        'outdir',
        '/data/vision/billf/mooncam/output/xiuming/reflectance/render/',
        "")
flags.DEFINE_float('diff_boost', 5, "")
flags.DEFINE_boolean('overwrite', False, "")
FLAGS = flags.FLAGS


def main(_):
    from IPython import embed; embed()
    scenes = (
        'lego',
        'ficus',
        'drums',
        'hotdog',
        'materials', # GI is problematic
    )[:4] # NOTE
    envmaps = (
        'probe_16-00_latlongmap', # good
        # '2159',
        # '2184',
        # '2234',
        # '2245',
        # '2250',
        # '3071',
        '3072',
        # '3083',
        # '3089',
        # '2144',
        '2188',
        '2163',
        # '2171',
    ) # NOTE
    vali_first_n = (
        None,
        8,
    )[1] # NOTE
    envmap_inten = 3
    res = 512
    spp = 128
    proj_root = '/data/vision/billf/intrinsic/sim/'
    tmpdir = join(proj_root, 'tmp/make_nerf_dataset')
    envmap_path = join(
        proj_root, 'data/envmaps/for-render_h16/train/{envmap}.hdr')
    cam_paths = join(proj_root, 'data/cams/nerf/transforms_*.json')
    test_envmap_dir = join(proj_root, 'data/envmaps/for-render_h16/test')
    id_ = '{mode}_{view_id:03d}'

    params_f = join(tmpdir, 'params.txt')
    expects_f = join(tmpdir, 'expects.txt')

    if not exists(tmpdir):
        makedirs(tmpdir)

    params_h = open(params_f, 'w')
    expects_h = open(expects_f, 'w')

    for scene in scenes:
        scene_path = join(proj_root, f'data/scenes/{scene}.blend')
        out_root = join(
            proj_root, f'data/render_outdoor_inten{envmap_inten}_gi',
            '%s_{envmap}' % scene) # NOTE

        for envmap in envmaps:
            envmap_path_ = envmap_path.format(envmap=envmap)
            out_root_ = out_root.format(envmap=envmap)

            # For training, validation, testing modes
            for cams_json in glob(cam_paths):
                mode = basename(cams_json)[:-5].split('_')[-1]

                # Load JSON
                data = xm.io.json.load(cams_json)
                cam_angle_x = data['camera_angle_x']
                frames = data['frames']

                if mode == 'val' and vali_first_n is not None:
                    frames = frames[:vali_first_n]

                # Correct the paths in JSON, to be JaxNeRF-compatible
                data = {'camera_angle_x': cam_angle_x, 'frames': []}
                for i, frame in enumerate(frames):
                    folder = id_.format(mode=mode, view_id=i)
                    frame['file_path'] = './%s/rgba' % folder
                    data['frames'].append(frame)
                json_path = join(out_root_, 'transforms_%s.json' % mode)
                xm.io.json.write(data, json_path)

                # For each frame
                for i, frame in enumerate(frames):
                    cam_transform_mat = frame['transform_matrix']
                    cam_transform_mat = listify_matrix(cam_transform_mat)
                    cam_transform_mat = ','.join(
                        str(x) for x in cam_transform_mat)

                    outdir = join(out_root_, id_.format(mode=mode, view_id=i))
                    rgba_out = join(outdir, 'rgba.png')
                    albedo_out = join(outdir, 'albedo.png')
                    normal_out = join(outdir, 'normal.png')
                    metadata_out = join(outdir, 'metadata.json')
                    # NOTE: relit renders are not so essential (we can just skip
                    # missing files in computing the scores with them), so we
                    # are not checking their existence

                    params_h.write((
                        f'--scene={scene_path} '
                        f'--cam_transform_mat={cam_transform_mat} '
                        f'--cam_angle_x={cam_angle_x} --envmap={envmap_path_} '
                        f'--envmap_inten={envmap_inten} '
                        f'--test_envmap_dir={test_envmap_dir} '
                        f'--imh={res} --imw={res} --spp={spp} '
                        f'--outdir={outdir}\n'))
                    expects_h.write(
                        f'{rgba_out} {albedo_out} {normal_out} {metadata_out}\n')

    params_h.close()
    expects_h.close()

    print(
        "For task parameter files, see\n\t{param_f}\n\t{expect_f}".format(
            param_f=params_f, expect_f=expects_f))


def listify_matrix(mat):
    elements = []
    for row in mat:
        for x in row:
            elements.append(x)
    return elements


if __name__ == '__main__':
    # Blender-Python binary
    argv = sys.argv
    if '--' in argv:
        arg_i = argv.index('--')
        argv = argv[(arg_i - 1):arg_i] + argv[(arg_i + 1):]

    app.run(main=main, argv=argv)
