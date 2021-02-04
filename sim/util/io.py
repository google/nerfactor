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

"""Lazy imports for XManager (which doesn't use the BUILD file).
"""

from os.path import dirname
from configparser import ConfigParser
import pickle as pkl
import json
import numpy as np

from google3.pyglib import gfile

from . import logging as logutil


logger = logutil.Logger(loggee="util/io")


def all_exist(path_dict):
    for _, v in path_dict.items():
        if not gfile.Exists(v):
            return False
    return True


def restore_model(model, ckpt_path):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    model.register_trainable()
    assert model.trainable_registered, (
        "Register the trainable layers to have them restored from the "
        "checkpoint")

    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(ckpt_path).expect_partial()


def read_config(path):
    config = ConfigParser()
    with gfile.Open(path, 'r') as h:
        config.read_file(h)
    return config


def prepare_outdir(outdir, overwrite=False, quiet=False):
    if gfile.IsDirectory(outdir):
        # Directory already exists
        if not quiet:
            logger.info("Output directory already exisits:\n\t%s", outdir)
        if overwrite:
            gfile.DeleteRecursively(outdir)
            if not quiet:
                logger.warn("Output directory wiped:\n\t%s", outdir)
        else:
            if not quiet:
                logger.info("Overwrite is off, so doing nothing")
            return
    gfile.MakeDirs(outdir)


def dump_dict_tensors(tdict, outpath):
    outdir = dirname(outpath)
    gfile.MakeDirs(outdir)
    with gfile.Open(outpath, 'wb') as h:
        pkl.dump(tdict, h)


def imwrite_tensor(tensor_uint, out_prefix):
    from PIL import Image

    for i in range(tensor_uint.shape[0]):
        out_path = out_prefix + '_%03d.png' % i
        gfile.MakeDirs(dirname(out_path))
        arr = tensor_uint[i, :, :, :].numpy()
        img = Image.fromarray(arr)
        with gfile.Open(out_path, 'wb') as h:
            img.save(h)


def write_video(frames, outpath, fps=12, gfs_user=None):
    from google3.learning.deepmind.video.python import video_api

    assert frames, "No frames"
    outdir = dirname(outpath)
    with gfile.AsUser(gfs_user):
        if not gfile.Exists(outdir):
            gfile.MakeDirs(outdir)
        with video_api.write(outpath, fps=fps) as h:
            for frame in frames:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                h.add_frame(frame)


def write_apng(imgs, labels, outprefix):
    from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm

    font_ttf = (
        '/cns/ok-d/home/gcam-eng/gcam/interns/xiuming/relight/data/fonts'
        '/open-sans/OpenSans-Regular.ttf')
    apng_f = outprefix + '.apng'
    xm.vis.video.make_apng(
        imgs, labels=labels, label_top_left_xy=(200, 200), font_size=100,
        font_color=(1, 1, 1), font_ttf=font_ttf, outpath=apng_f)
    # Renaming for easier insertion into slides
    png_f = apng_f.replace('.apng', '.png')
    gfile.Rename(apng_f, png_f, overwrite=True)
    return png_f


def read_json(path):
    with gfile.Open(path, 'r') as h:
        data = json.load(h)
    return data


def write_json(data, path):
    out_dir = dirname(path)
    if not gfile.Exists(out_dir):
        gfile.MakeDirs(out_dir)

    with gfile.Open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)


def load_sfm_cam(cam_f):
    from google3.net.proto2.python.public import text_format
    from google3.vision.sfm import camera_pb2
    from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera

    if cam_f.endswith('.textproto'):
        with gfile.Open(cam_f) as h:
            proto = text_format.Parse(h.read(), camera_pb2.CameraProto())

    elif cam_f.endswith('.pb'):
        with gfile.Open(cam_f, 'rb') as h:
            proto = camera_pb2.CameraProto.FromString(h.read())

    else:
        raise NotImplementedError(cam_f)

    cam = vision_sfm_camera.Camera.FromProto(proto)
    return cam


def load_exr(exr_f):
    from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
    # requires the `xiuminglib:exr` rule

    exr = xm.io.exr.EXR(exr_path=exr_f)
    return exr.data


def load_np(np_f):
    if np_f.endswith('.npy'):
        with gfile.Open(np_f, 'rb') as h:
            data = np.load(h)
        return data

    # .npz
    with gfile.Open(np_f, 'rb') as h:
        data = np.load(h, allow_pickle=True)
        data = dict(data)
    return data
