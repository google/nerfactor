from os.path import join, basename
from absl import app, flags
import numpy as np
from tqdm import tqdm
import cv2
import trimesh

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.util.geom import write_xyz, write_lvis, write_alpha, write_normal


flags.DEFINE_string('cam_dir', '', "")
flags.DEFINE_string('surf_dir', '', "")
flags.DEFINE_string('img_dir', '', "")
flags.DEFINE_string('outdir', '', "")
flags.DEFINE_integer('h', 256, "")
flags.DEFINE_integer('light_h', 16, "")
flags.DEFINE_integer('n_vali', 2, "")
flags.DEFINE_integer('n_test', 120, "")
flags.DEFINE_float('lvis_eps', 1e-1, "")
flags.DEFINE_float('lvis_radius', 1e5, "")
flags.DEFINE_integer('lvis_fps', 12, "")
flags.DEFINE_boolean('debug', False, "debug toggle")
flags.DEFINE_boolean('overwrite', False, "overwrite toggle")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.outdir, rm_if_exists=FLAGS.overwrite)

    # Glob poses
    cam_paths = xm.os.sortglob(FLAGS.cam_dir, filename='pos_???', ext='txt')

    # Glob images
    ext = 'png'
    img_paths = xm.os.sortglob( # the most diffuse lighting
        FLAGS.img_dir, filename='*_3_*', ext=ext)
    assert img_paths, "No image globbed"

    # In case only the first 49 cameras are used to capture images
    cam_paths = cam_paths[:len(img_paths)]

    if FLAGS.debug:
        img_paths = img_paths[:4]
        cam_paths = cam_paths[:4]

    # Sanity check
    n_poses = len(cam_paths)
    n_imgs = len(img_paths)
    assert n_poses == n_imgs, (
        "Mismatch between numbers of images ({n_imgs}) and "
        "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)

    # Training-validation split
    ind_vali = np.arange(n_imgs)[:-1:(n_imgs // FLAGS.n_vali)]
    ind_train = np.array(
        [x for x in np.arange(n_imgs) if x not in ind_vali])

    # Load mesh
    scene = basename(FLAGS.img_dir)
    bn1 = basename(FLAGS.surf_dir) + '%03d' % int(scene.lstrip('scan'))
    bn2 = '_l3_surf_11_trim_8.ply'
    mesh_path = join(FLAGS.surf_dir, bn1 + bn2)
    mesh = trimesh.load(mesh_path)
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # Light locations
    light_w = 2 * FLAGS.light_h
    # DTU scenes are roughly 1000x bigger than (normalized) real NeRF scenes
    lxyzs, lareas = gen_light_xyz(
        FLAGS.light_h, light_w, envmap_radius=FLAGS.lvis_radius)
    # Shift the hemisphere to the scene center since centers of DTU scenes
    # are not at origin
    mesh_center = np.mean(mesh.vertices, axis=0)
    lxyzs += mesh_center
    # NOTE: DTU scenes have the z-axis flipped, so we flip the z sign
    lxyzs[:, :, 2] = -lxyzs[:, :, 2]
    # Save this because this is scene-specific and hence cannot be generated
    # on the fly during training (like other non-DTU scenes)
    lights = {'lxyzs': lxyzs, 'lareas': lareas}
    lights_npz = join(FLAGS.outdir, 'lights.npz')
    np.savez(lights_npz, **lights)
    #
    lxyzs_flat = np.reshape(lxyzs, (-1, 3))

    imgs, cams = [], []
    factor = None
    train_i, vali_i = 0, 0
    for i, (img_path, cam_path) in enumerate(
            tqdm(
                zip(img_paths, cam_paths), desc="Train. & Vali.",
                total=n_imgs)):
        # Load and resize image
        img = xm.io.img.read(img_path)
        img = xm.img.normalize_uint(img)
        if factor is None:
            factor = float(img.shape[0]) / FLAGS.h
        else:
            assert float(img.shape[0]) / FLAGS.h == factor, \
                "Images are of varying sizes"
        img = xm.img.resize(img, new_h=FLAGS.h, method='tf')
        if img.shape[2] == 3:
            # NOTE: add an all-one alpha
            img = np.dstack((img, np.ones_like(img)[:, :, :1]))
        imgs.append(img)

        # Pose
        P = np.loadtxt(cam_path)
        K = cv2.decomposeProjectionMatrix(P)[0]
        Rt = np.linalg.inv(K).dot(P) # o2cvc
        K = K / K[2, 2]
        f = (K[0, 0] + K[1, 1]) / 2 # hacky but need a single focal length
        f *= 1. / factor # scale according to the new resolution
        K = np.array([
            [f, 0, img.shape[1] / 2], [0, f, img.shape[0] / 2], [0, 0, 1]])

        # Cast rays from this camera to the geometry
        cam = xm.camera.PerspCam() # a CV camera "cvc"
        cam.int_mat = K
        cam.ext_mat = Rt
        ray_dirs = cam.gen_rays()
        hwn = ray_dirs.shape[:3]
        ray_dirs_flat = np.reshape(ray_dirs, (-1, 3))
        n_rays = ray_dirs_flat.shape[0]
        ray_origs = np.tile(cam.loc, (n_rays, 1))
        locs, ray_ind, tri_ind = inter.intersects_location(
            ray_origs, ray_dirs_flat, multiple_hits=False)
        cams.append(cam)

        if i in ind_train:
            view_name = f'train_{train_i:03d}'
            train_i += 1
        else:
            view_name = f'val_{vali_i:03d}'
            vali_i += 1
        #    src = join(FLAGS.outdir, 'vali' + view_name[3:])
        #    from os.path import exists
        #    if exists(src):
        #        tgt = join(FLAGS.outdir, view_name)
        #        import shutil
        #        shutil.move(src, tgt)
        outdir = join(FLAGS.outdir, view_name)

        # Metadata
        metadata = {
            'id': view_name, 'imh': hwn[0], 'imw': hwn[1],
            'cam_loc': cam.loc.tolist()}
        meta_json = join(outdir, 'metadata.json')
        xm.io.json.write(metadata, meta_json)

        # RGB
        rgba_png = join(outdir, 'rgba.png')
        xm.io.img.write_float(img, rgba_png, clip=True)

        alpha = gen_alpha(ray_ind, hwn, outdir)
        xyz = gen_xyz(ray_ind, locs, hwn, outdir)
        normal = gen_normal(ray_ind, tri_ind, mesh, hwn, outdir)
        #gen_lvis(xyz, lxyzs_flat, inter, normal, alpha, outdir)

    # ------ Test Data

    cam_locs = [x.loc for x in cams]
    cam_dists = [np.linalg.norm(x - mesh_center) for x in cam_locs]
    cam_dist = 1.5 * np.mean(cam_dists)

    lngs = np.linspace(-0.25 * np.pi, 0.5 * np.pi, FLAGS.n_test // 2)
    lngs = np.hstack((
        lngs,
        np.linspace(0.5 * np.pi, -0.25 * np.pi, FLAGS.n_test - len(lngs))))
    lats = np.linspace(-0.25 * np.pi, 0, FLAGS.n_test)

    if FLAGS.debug:
        lngs = lngs[:4]
        lats = lats[:4]

    frames = []
    for i, (lat, lng) in enumerate(
            tqdm(zip(lats, lngs), desc="Test", total=FLAGS.n_test)):
        # Test camera
        cam_loc = xm.geometry.sph.sph2cart( # centered at (0, 0, 0)
            (cam_dist, lat, lng), convention='lat-lng')
        cam_loc += mesh_center
        cam.loc = cam_loc
        cam.lookat = mesh_center
        cam.up = (0, 0, -1) # up is -z

        # Cast rays from camera to object
        ray_dirs = cam.gen_rays()
        hwn = ray_dirs.shape[:3]
        ray_dirs_flat = np.reshape(ray_dirs, (-1, 3))
        n_rays = ray_dirs_flat.shape[0]
        ray_origs = np.tile(cam.loc, (n_rays, 1))
        locs, ray_ind, tri_ind = inter.intersects_location(
            ray_origs, ray_dirs_flat, multiple_hits=False)

        view_name = f'test_{i:03d}'
        outdir = join(FLAGS.outdir, view_name)

        # Metadata
        metadata = {
            'id': view_name, 'imh': hwn[0], 'imw': hwn[1],
            'cam_loc': cam.loc.tolist()}
        meta_json = join(outdir, 'metadata.json')
        xm.io.json.write(metadata, meta_json)

        # NN
        dist_to_train = [np.linalg.norm(cam.loc - x) for x in cam_locs]
        nn = imgs[np.argmin(dist_to_train)]
        nn_png = join(outdir, 'nn.png')
        xm.io.img.write_float(nn, nn_png, clip=True)

        # Real stuff
        alpha = gen_alpha(ray_ind, hwn, outdir)
        xyz = gen_xyz(ray_ind, locs, hwn, outdir)
        normal = gen_normal(ray_ind, tri_ind, mesh, hwn, outdir)
        #gen_lvis(xyz, lxyzs_flat, inter, normal, alpha, outdir)

        # Visualization frames
        if FLAGS.debug:
            normal_img = np.clip((normal + 1) / 2, 0, 1)
            frame = xm.img.denormalize_float(normal_img)
            text = "lat: {lat:.2f}; lng: {lng:.2f}".format(
                lat=lat / np.pi * 180, lng=lng / np.pi * 180)
            frame = xm.vis.text.put_text(frame, text)
            frames.append(frame)
    if FLAGS.debug:
        xm.vis.video.make_video(frames)


def gen_alpha(ray_ind, hwn, outdir):
    """0 means background, and 1 means foreground.
    """
    n_rays = np.prod(hwn)
    alpha_flat = np.zeros((n_rays,))

    alpha_flat[ray_ind] = 1
    alpha = np.reshape(alpha_flat, hwn)

    alpha = np.mean(alpha, axis=2) # average over samples

    #kernel = np.ones(3, np.uint8)
    #alpha = cv2.erode(alpha, kernel, iterations=10)

    write_alpha(alpha, outdir)
    return alpha


def gen_xyz(ray_ind, locs, hwn, outdir):
    """XYZ buffer has filling value as (0, 0, 0).
    """
    n_rays = np.prod(hwn)
    xyz_flat = np.zeros((n_rays, 3))

    xyz_flat[ray_ind] = locs
    xyz = np.reshape(xyz_flat, hwn + (3,))

    xyz = np.mean(xyz, axis=2) # average over samples

    write_xyz(xyz, outdir)
    return xyz


def gen_normal(ray_ind, tri_ind, mesh, hwn, outdir):
    """Normal buffer has filling value as (0, 1, 0).
    """
    n_rays = np.prod(hwn)
    normal_flat = np.zeros((n_rays, 3))
    normal_flat[:, 1] = 1

    hit_normals = mesh.face_normals[tri_ind]
    normal_flat[ray_ind] = hit_normals
    normal = np.reshape(normal_flat, hwn + (3,))

    normal = np.mean(normal, axis=2) # average over samples
    normal = xm.linalg.normalize(normal, axis=2) # re-normalize

    write_normal(normal, outdir)
    return normal


def gen_lvis(xyz, lxyzs_flat, inter, normal, alpha, outdir):
    """Light visibility buffers.
    """
    ray_origs = np.tile(
        xyz[:, :, None, :], (1, 1, lxyzs_flat.shape[0], 1)) # HxWxLx3
    ray_dirs = lxyzs_flat[None, None, :, :] - ray_origs # HxWxLx3
    ray_dirs = xm.linalg.normalize(ray_dirs, axis=3)
    ray_origs_flat = np.reshape(ray_origs, (-1, 3))
    ray_dirs_flat = np.reshape(ray_dirs, (-1, 3))

    # Move away from the surface a bit to avoid numerical issues
    ray_origs_flat += ray_dirs_flat * FLAGS.lvis_eps

    # Cast rays
    _, ray_ind, _ = inter.intersects_location(
        ray_origs_flat, ray_dirs_flat, multiple_hits=False)
    lviss_flat = np.ones((np.prod(ray_dirs.shape[:3]),)) # HWL
    lviss_flat[ray_ind] = 0 # hit means blocked
    lviss = np.reshape(lviss_flat, ray_dirs.shape[:3]) # HxWxL

    # Negative normals also lead to zero visibility
    cos = np.einsum('ijl,ijkl->ijk', normal, ray_dirs)
    lviss[cos <= 0] = 0

    # Mask out background
    lviss *= alpha[:, :, None]

    write_lvis(lviss, FLAGS.lvis_fps, outdir)


if __name__ == '__main__':
    app.run(main=main)
