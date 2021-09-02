from os.path import join, basename
from absl import app, flags
import numpy as np
from tqdm import tqdm
import cv2
import trimesh

# from third_party.xiuminglib import xiuminglib as xm
import xiuminglib as xm
from data_gen.util import gen_data


flags.DEFINE_string('cam_dir', '', "")
flags.DEFINE_string('surf_dir', '', "")
flags.DEFINE_string('img_root', '', "")
flags.DEFINE_string('img_outroot', '', "")
flags.DEFINE_string('surf_outroot', '', "")
flags.DEFINE_list('scenes', [], "")
flags.DEFINE_integer('h', 256, "")
flags.DEFINE_integer('n_vali', 2, "")
flags.DEFINE_boolean('debug', False, "debug toggle")
flags.DEFINE_boolean('overwrite', False, "overwrite toggle")
FLAGS = flags.FLAGS


def main(_):
    #xm.os.makedirs(FLAGS.img_outroot, rm_if_exists=FLAGS.overwrite)
    xm.os.makedirs(FLAGS.surf_outroot, rm_if_exists=FLAGS.overwrite)

    for scene in tqdm(FLAGS.scenes, desc="Scenes"):
        # Glob poses
        cam_paths = xm.os.sortglob(FLAGS.cam_dir, filename='pos_???', ext='txt')

        # Glob images
        img_dir = join(FLAGS.img_root, scene)
        ext = 'png'
        img_paths = xm.os.sortglob( # the most diffuse lighting
            img_dir, filename='*_3_*', ext=ext)
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

        # Load mesh
        bn1 = basename(FLAGS.surf_dir) + '%03d' % int(scene.lstrip('scan'))
        bn2 = '_l3_surf_11.ply'
        mesh_path = join(FLAGS.surf_dir, bn1 + bn2)
        mesh = trimesh.load(mesh_path)
        inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        poses, imgs = [], []
        factor = None
        for img_path, cam_path in tqdm(
                zip(img_paths, cam_paths), desc="Converting", total=n_imgs):
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
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3] # w2c
            K = K / K[2, 2]
            f = (K[0, 0] + K[1, 1]) / 2 # hacky but need a single focal length
            f *= 1. / factor # scale according to the new resolution
            #
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose() # c2w
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            #
            coord_trans_world = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            coord_trans_cam = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pose = coord_trans_world.dot(pose).dot(coord_trans_cam)
            # Camera-to-world (OpenGL)
            hw = np.array(img.shape[:2]).reshape([2, 1])
            hwf = np.vstack((hw, [f]))
            pose = np.hstack((pose[:3, :], hwf))
            poses.append(pose)

            # Cast rays from this camera to the geometry
            int_mat = np.array([
                [f, 0, hw[1, 0] / 2], [0, f, hw[0, 0] / 2], [0, 0, 1]])
            ext_mat = np.hstack([R, t[:3] / t[3]])
            cam = xm.camera.PerspCam()
            cam.int_mat = int_mat
            cam.ext_mat = ext_mat
            ray_dirs = cam.gen_rays()
            ray_dirs_flat = np.reshape(ray_dirs, (-1, 3))
            ray_origs = np.tile(cam.loc, (ray_dirs_flat.shape[0], 1))
            locs, ray_ind, tri_ind = inter.intersects_location(
                ray_origs, ray_dirs_flat, multiple_hits=False)
            xyz_flat = np.zeros((ray_dirs_flat.shape[0], 3))
            xyz_flat[ray_ind] = locs
            xyz = np.reshape(xyz_flat, ray_dirs.shape)
            xyz = np.mean(xyz, axis=2)
            from IPython import embed; embed()
        imgs = np.stack(imgs, axis=-1)
        poses = np.dstack(poses) # 3x5xN

        # Move variable dim to axis 0
        poses = np.moveaxis(poses, -1, 0).astype(np.float32) # Nx3x5
        imgs = np.moveaxis(imgs, -1, 0) # NxHxWx4

        #outdir = join(FLAGS.img_outroot, scene)
        #gen_data(poses, imgs, img_paths, FLAGS.n_vali, outdir)


if __name__ == '__main__':
    app.run(main=main)
