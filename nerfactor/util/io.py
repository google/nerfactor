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

from os import makedirs
from os.path import dirname, exists, isdir
from shutil import rmtree
from configparser import ConfigParser
import pickle as pkl
import json
import numpy as np

from . import logging as logutil


logger = logutil.Logger(loggee="util/io")


def all_exist(path_dict):
    for _, v in path_dict.items():
        if not exists(v):
            return False
    return True


def restore_model(model, ckpt_path):
    import tensorflow as tf

    model.register_trainable()
    assert model.trainable_registered, (
        "Register the trainable layers to have them restored from the "
        "checkpoint")

    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(ckpt_path).expect_partial()


def read_config(path):
    config = ConfigParser()
    with open(path, 'r') as h:
        config.read_file(h)
    return config


def write_config(config, path):
    with open(path, 'w') as h:
        config.write(h)


def prepare_outdir(outdir, overwrite=False, quiet=False):
    if isdir(outdir):
        # Directory already exists
        if not quiet:
            logger.info("Output directory already exisits:\n\t%s", outdir)
        if overwrite:
            rmtree(outdir)
            if not quiet:
                logger.warn("Output directory wiped:\n\t%s", outdir)
        else:
            if not quiet:
                logger.info("Overwrite is off, so doing nothing")
            return
    makedirs(outdir)


def dump_dict_tensors(tdict, outpath):
    outdir = dirname(outpath)
    makedirs(outdir)
    with open(outpath, 'wb') as h:
        pkl.dump(tdict, h)


def imwrite_tensor(tensor_uint, out_prefix):
    from PIL import Image

    for i in range(tensor_uint.shape[0]):
        out_path = out_prefix + '_%03d.png' % i
        makedirs(dirname(out_path))
        arr = tensor_uint[i, :, :, :].numpy()
        img = Image.fromarray(arr)
        with open(out_path, 'wb') as h:
            img.save(h)


def write_video(frames, outpath, fps=12):
    assert frames, "No frames"
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)
    raise NotImplementedError # FIXME


def read_json(path):
    with open(path, 'r') as h:
        data = json.load(h)
    return data


def write_json(data, path):
    out_dir = dirname(path)
    if not exists(out_dir):
        makedirs(out_dir)

    with open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)


def load_exr(exr_f):
    import xiuminglib as xm

    exr = xm.io.exr.EXR(exr_path=exr_f)
    return exr.data


def load_np(np_f):
    if np_f.endswith('.npy'):
        with open(np_f, 'rb') as h:
            data = np.load(h)
        return data

    # .npz
    with open(np_f, 'rb') as h:
        data = np.load(h, allow_pickle=True)
        data = dict(data)
    return data
