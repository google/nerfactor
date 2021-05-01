#!/usr/bin/env python

import sys
from os import makedirs
from os.path import join, exists, basename
from glob import glob
from absl import app, flags
import xiuminglib as xm


flags.DEFINE_string('scene_blend', '', "path to the Blender scene")
flags.DEFINE_string('light_path', '', "path to the light probe")
flags.DEFINE_string('cam_dir', '', "directory containing the camera JSONs")
flags.DEFINE_string(
    'test_light_dir', '', "directory containing the test (novel) light probes")
flags.DEFINE_integer('vali_first_n', 8, "")
flags.DEFINE_float('light_inten', 3, "global scale for the light probe")
flags.DEFINE_integer('res', 512, "resolution of the squre renders")
flags.DEFINE_integer('spp', 128, "samples per pixel")
flags.DEFINE_string('outdir', '', "output directory")
flags.DEFINE_boolean('overwrite', False, "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.outdir, rm_if_exists=FLAGS.overwrite)

    # For training, validation, testing modes
    for cams_json in xm.os.sortglob(FLAGS.cam_dir, '*', ext='json'):
        mode = basename(cams_json)[:-len('.json')].split('_')[-1]
        print(mode)

        # Load JSON
        data = xm.io.json.load(cams_json)
        cam_angle_x = data['camera_angle_x']
        frames = data['frames']
        from IPython import embed; embed()

        if mode == 'val' and vali_first_n is not None:
            frames = frames[:vali_first_n]

        # Correct the paths in JSON, to be JaxNeRF-compatible
        data = {'camera_angle_x': cam_angle_x, 'frames': []}
        for i, frame in enumerate(frames):
            folder = f'{mode}_{i:03d}'
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

            outdir = join(out_root_, f'{mode}_{i:03d}')
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
                f'--light_inten={light_inten} '
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
