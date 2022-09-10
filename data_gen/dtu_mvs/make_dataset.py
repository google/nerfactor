from os.path import join, basename
from absl import app, flags
import numpy as np
from tqdm import tqdm
import cv2

# from third_party.xiuminglib import xiuminglib as xm
import xiuminglib as xm
from data_gen.util import gen_data


flags.DEFINE_string('scene_dir', '', "scene directory")
flags.DEFINE_integer('h', 256, "output image height")
flags.DEFINE_integer('n_vali', 2, "number of held-out validation views")
flags.DEFINE_string('outroot', '', "output root")
flags.DEFINE_boolean('debug', False, "debug toggle")
flags.DEFINE_boolean('overwrite', False, "overwrite toggle")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.outroot, rm_if_exists=FLAGS.overwrite)

    # Load poses
    cams_npz = join(FLAGS.scene_dir, 'cameras.npz')
    cams = xm.io.np.read_or_write(cams_npz)

    # Glob images
    img_dir = join(FLAGS.scene_dir, 'image')
    ext = 'png'
    img_paths = xm.os.sortglob(img_dir, ext=ext)
    assert img_paths, "No image globbed"

    poses, imgs = [], []
    factor = None
    for img_path in tqdm(img_paths, desc="Loading Images"):
        # Load and resize images
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

        # Poses
        i = int(basename(img_path).rstrip('.' + ext))
        world_mat = cams[f'world_mat_{i}']
        scale_mat = cams[f'scale_mat_{i}']
        #
        P = world_mat[:3]
        K, R, t = cv2.decomposeProjectionMatrix(P)[:3] # w2c
        K = K / K[2, 2]
        f = (K[0, 0] + K[1, 1]) / 2 # hacky but we need a single focal length
        #
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose() # c2w
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        #
        norm_trans = scale_mat[:3, 3:]
        norm_scale = np.diagonal(scale_mat[:3, :3])[..., None]
        pose[:3, 3:] -= norm_trans
        pose[:3, 3:] /= norm_scale
        #
        coord_trans_world = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        coord_trans_cam = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pose = coord_trans_world.dot(pose).dot(coord_trans_cam)
        # Camera-to-world (OpenGL)
        hw = np.array(img.shape[:2]).reshape([2, 1])
        hwf = np.vstack((hw, [f * 1. / factor]))
        pose = np.hstack((pose[:3, :], hwf))
        poses.append(pose)
    imgs = np.stack(imgs, axis=-1)
    poses = np.dstack(poses) # 3x5xN

    if FLAGS.debug:
        imgs = imgs[..., :4]
        poses = poses[..., :4]

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, (
        "Mismatch between numbers of images ({n_imgs}) and "
        "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)

    # Move variable dim to axis 0
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # Nx3x5
    imgs = np.moveaxis(imgs, -1, 0) # NxHxWx4

    gen_data(poses, imgs, img_paths, FLAGS.n_vali, FLAGS.outroot)


if __name__ == '__main__':
    app.run(main=main)
