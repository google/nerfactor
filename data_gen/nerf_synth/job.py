from sys import argv
from argparse import ArgumentParser
from os import makedirs
from os.path import join, basename, exists
import numpy as np
import cv2

# Blender
import bpy
from mathutils import Matrix

import xiuminglib as xm

import util


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
parser.add_argument(
    '--cam_transform_mat', type=str, required=True, help="")
parser.add_argument(
    '--cam_angle_x', type=float, required=True, help="")
parser.add_argument(
    '--envmap', type=str, required=True, help="path to the environment map")
parser.add_argument(
    '--envmap_inten', type=float, default=1,
    help="environment lighting's global intensity scale")
parser.add_argument(
    '--test_envmap_dir', type=str, default=None,
    help="directory to additional test HDR maps or JSONs (for OLAT)")
parser.add_argument('--imh', type=int, default=256, help="image height")
parser.add_argument('--imw', type=int, default=256, help="image width")
parser.add_argument(
    '--spp', type=int, default=64, help="samples per pixel for rendering")
parser.add_argument(
    '--outdir', type=str, required=True, help="output directory")
parser.add_argument('--direct_only', type=bool, default=False, help="")
parser.add_argument('--debug', type=bool, default=False, help="")


def main(args):
    if not exists(args.outdir):
        makedirs(args.outdir)

    # Dump metadata
    metadata_json = join(args.outdir, 'metadata.json')
    data = {
        'scene': basename(args.scene),
        'cam_transform_mat': args.cam_transform_mat,
        'cam_angle_x': args.cam_angle_x, 'envmap': basename(args.envmap),
        'envmap_inten': args.envmap_inten, 'imh': args.imh, 'imw': args.imw,
        'spp': args.spp}
    xm.io.json.write(data, metadata_json)

    # Open scene
    xm.blender.scene.open_blend(args.scene)

    # Remove empty tracker that may mess up the camera pose
    objs = [
        x for x in bpy.data.objects if x.type == 'EMPTY' and 'Empty' in x.name]
    bpy.ops.object.delete({'selected_objects': objs})

    # Remove undesired objects
    objs = []
    for o in bpy.data.objects:
        if o.name == 'BackgroundPlane':
            objs.append(o)
    bpy.ops.object.delete({'selected_objects': objs})

    # Remove existing lights, if any
    objs = []
    for o in bpy.data.objects:
        if o.type == 'LIGHT':
            objs.append(o)
        elif o.active_material is not None:
            for node in o.active_material.node_tree.nodes:
                if node.type == 'EMISSION':
                    objs.append(o)
    bpy.ops.object.delete({'selected_objects': objs})

    # Set camera
    cam_obj = bpy.data.objects['Camera']
    cam_obj.data.sensor_width = args.imw
    cam_obj.data.sensor_height = args.imh
    # 1 pixel is 1 mm
    cam_obj.data.lens = .5 * args.imw / np.tan(.5 * args.cam_angle_x)
    cam_transform_mat = [float(x) for x in args.cam_transform_mat.split(',')]
    cam_transform_mat = np.array(cam_transform_mat).reshape((4, 4))
    cam_transform_mat = Matrix(cam_transform_mat)
    # NOTE: If not wrapping the NumPy array as a Matrix, it would be transposed
    # for some unknown reason: https://blender.stackexchange.com/q/159824/30822
    cam_obj.matrix_world = cam_transform_mat
    bpy.context.view_layer.update()

    # Add environment lighting
    xm.blender.light.add_light_env(env=args.envmap, strength=args.envmap_inten)

    # Rendering settings
    xm.blender.render.easyset(w=args.imw, h=args.imh, n_samples=args.spp)
    # FIXME
    # if args.direct_only:
    #     bpy.context.scene.cycles.max_bounces = 0

    # Render RGBA
    rgba_png = join(args.outdir, 'rgba.png')
    xm.blender.render.render(rgba_png, cam=cam_obj)
    rgba = cv2.imread(rgba_png, cv2.IMREAD_UNCHANGED)
    alpha = rgba[:, :, 3]
    alpha = xm.img.normalize_uint(alpha)

    if args.debug:
        snapshot_path = join(args.outdir, 'scene.blend')
        xm.blender.scene.save_blend(snapshot_path)

    # Render relit ground truth
    if args.test_envmap_dir is not None:
        # With HDR maps
        for envmap_path in xm.os.sortglob(args.test_envmap_dir, '*.hdr'):
            envmap_name = basename(envmap_path).split('.')[0]
            xm.blender.light.add_light_env(env=envmap_path, strength=1.)
            outpath = join(args.outdir, 'rgba_%s.png' % envmap_name)
            xm.blender.render.render(outpath, cam=cam_obj)
        # With OLAT
        for envmap_path in xm.os.sortglob(args.test_envmap_dir, '*.json'):
            envmap_name = basename(envmap_path).split('.')[0]
            olat = xm.io.json.load(envmap_path)
            # NOTE: not using intensity in JSON; because Blender uses Watts
            # (and fall-off), it's impossible to match exactly our predictions
            xm.blender.light.add_light_env(
                env=(1, 1, 1, 1), strength=0) # ambient
            pt_light = xm.blender.light.add_light_point( # point
                xyz=olat['point_location'], energy=50_000)
            outpath = join(args.outdir, 'rgba_%s.png' % envmap_name)
            xm.blender.render.render(outpath, cam=cam_obj)
            xm.blender.object.remove_objects(pt_light.name) # avoid light accu.

    # Render albedo
    # Let's assume white specularity, so the diffuse_color alone is albedo
    diffuse_color_exr = join(args.outdir, 'diffuse-color.exr')
    # glossy_color_exr = join(args.outdir, 'glossy_color.exr')
    xm.blender.render.render_lighting_passes(
        diffuse_color_exr, cam=cam_obj, select='diffuse_color')
    # xm.blender.render.render_lighting_passes(
    #     glossy_color_exr, cam=cam_obj, select='glossy_color')
    diffuse_color = util.read_exr(diffuse_color_exr)
    # glossy_color = util.read_exr(glossy_color_exr)
    albedo = diffuse_color # + glossy_color
    albedo = np.dstack((albedo, alpha))
    albedo_png = join(args.outdir, 'albedo.png')
    xm.io.img.write_arr(albedo, albedo_png)

    # Render normals ...
    normal_exr = join(args.outdir, 'normal.exr')
    normal_refball_exr = join(args.outdir, 'refball-normal.exr')
    xm.blender.render.render_normal(
        normal_exr, cam=cam_obj, world_coords=True,
        outpath_refball=normal_refball_exr)
    normals = util.read_exr(normal_exr)
    normal_png = join(args.outdir, 'normal.png')
    xm.vis.geometry.normal_as_image(
        normals, alpha, outpath=normal_png, keep_alpha=True)
    # and also normals of the reference ball
    normals_refball = util.read_exr(normal_refball_exr)
    normal_refball_png = normal_refball_exr[:-len('.exr')] + '.png'
    xm.vis.geometry.normal_as_image(
        normals_refball, outpath=normal_refball_png, keep_alpha=True)


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
