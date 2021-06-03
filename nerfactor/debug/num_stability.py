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

from os.path import join, basename, dirname
import numpy as np
from tqdm import tqdm
from absl import app, flags
import tensorflow as tf

from nerfactor import datasets, models
from nerfactor.util import io as ioutil, logging as logutil, \
    math as mathutil, geom as geomutil
from brdf.microfacet.microfacet import Microfacet
from third_party.xiuminglib import xiuminglib as xm


FLAGS = flags.FLAGS

logger = logutil.Logger()


def microfacet():
    normal = tf.convert_to_tensor((0, 0, 1), dtype=tf.float32)
    pts2c = tf.convert_to_tensor((1, 1, 1), dtype=tf.float32)
    pts2l = tf.convert_to_tensor((-1, -1, 1), dtype=tf.float32)
    albedo = tf.convert_to_tensor((1, 1, 1), dtype=tf.float32)
    # rough = tf.linspace(0, 1, 10)
    # rough = tf.convert_to_tensor(
    #     (-1e-12, -1e-24, 0, 1e-24, 1e-12, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    #     dtype=tf.float32)
    # rough = tf.linspace(0., 0.05, 50)
    # rough = tf.linspace(0.001, 0.01, 10)
    # rough = tf.linspace(0.01, 0.1, 10)
    rough = tf.linspace(0.1, 1, 10)
    rough = tf.cast(rough, tf.float32)

    # Repeating other directions to match number of roughness values
    rough = tf.reshape(rough, (rough.shape[0], 1))
    normal = tf.tile(normal[None, :], (rough.shape[0], 1))
    pts2c = tf.tile(pts2c[None, :], (rough.shape[0], 1))
    pts2l = tf.tile(pts2l[None, :], (rough.shape[0], 1))
    pts2l = tf.reshape(pts2l, (pts2l.shape[0], 1, 3))
    albedo = tf.tile(albedo[None, :], (rough.shape[0], 1))

    # Normalizing
    pts2c = tf.linalg.l2_normalize(pts2c, axis=1)
    pts2l = tf.linalg.l2_normalize(pts2l, axis=2)
    normal = tf.linalg.l2_normalize(normal, axis=1)

    brdf = Microfacet()

    with tf.GradientTape() as tape1:
        tape1.watch(rough)
        y1 = brdf(pts2l, pts2c, normal, albedo=albedo, rough=rough)
    g1 = tape1.gradient(y1, rough)
    print(g1)


def sqrt():
    x = tf.convert_to_tensor((-1e24, -1e-12, 0, 1e-12), dtype=tf.float32)

    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y1 = tf.sqrt(x)
    g1 = tape1.gradient(y1, x)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = tf.sqrt(x)
    g2 = tape2.gradient(y2, x)

    print("Input:\n", x)
    print("Forward:\n", y1, y2)
    print("Backward:\n", g1, g2)
    from IPython import embed; embed()


def divide_no_nan():
    x = tf.convert_to_tensor((-1e24, 0, 1e-12, 1e6, 1e24), dtype=tf.float32)

    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y1 = 1. / x
    g1 = tape1.gradient(y1, x)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = tf.math.divide_no_nan(1., x)
    g2 = tape2.gradient(y2, x)

    print("Input:\n", x)
    print("Forward:\n", y1, y2)
    print("Backward:\n", g1, g2)
    from IPython import embed; embed()


def acos():
    x = tf.convert_to_tensor((-1 - 1e-6, 1 + 1e-6), dtype=tf.float32)
    x = tf.concat((x, tf.linspace(-1., 1., 8)), 0)

    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y1 = tf.acos(x)
    g1 = tape1.gradient(y1, x)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = mathutil.safe_acos(x)
    g2 = tape2.gradient(y2, x)

    print("Input:\n", x)
    print("Forward:\n", y1, y2)
    print("Backward:\n", g1, g2)
    from IPython import embed; embed()


def atan2():
    x = tf.convert_to_tensor((
        (0, 0),
        (1, 0),
        (0, 1),
    ), dtype=tf.float32)
    x_pp = tf.random.uniform((2, 2), dtype=tf.float32)
    x_nn = -tf.random.uniform((2, 2), dtype=tf.float32)
    x_pn = tf.concat((
        tf.random.uniform((2, 1), dtype=tf.float32),
        -tf.random.uniform((2, 1), dtype=tf.float32)), axis=1)
    x_np = tf.concat((
        -tf.random.uniform((2, 1), dtype=tf.float32),
        tf.random.uniform((2, 1), dtype=tf.float32)), axis=1)
    x = tf.concat((x, x_pp, x_nn, x_pn, x_np), axis=0)

    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y1 = tf.atan2(x[:, 0], x[:, 1])
    g1 = tape1.gradient(y1, x)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = mathutil.safe_atan2(x[:, 0], x[:, 1])
    g2 = tape2.gradient(y2, x)

    print("Forward:\n", y1, y2)
    print("Backward:\n", g1, g2)


def dir2rusink():
    ldir = tf.convert_to_tensor(
        (-0.7221278, 0.19019903, -0.66509825), dtype=tf.float32)
    #vdir = tf.convert_to_tensor(
    #    (-0.72259355, 0.18995164, -0.66466284), dtype=tf.float32)
    vdir = ldir
    ldir = tf.convert_to_tensor(
        [-0.95697343, -0.27924237, 0.07890151], dtype=tf.float32)
    vdir = tf.convert_to_tensor(
        [-0.9570226, -0.27924448, 0.07829469], dtype=tf.float32)

    ldir = tf.reshape(ldir, (1, 3))
    vdir = tf.reshape(vdir, (1, 3))

    with tf.GradientTape() as tape:
        tape.watch(ldir)
        rusink = geomutil.dir2rusink(ldir, vdir)

    g = tape.gradient(rusink, ldir)
    from IPython import embed; embed()


def gen_world2local():
    eps = 1e-4

    normal = tf.random.uniform((1024, 3), dtype=tf.float32)
    normal = tf.convert_to_tensor((
        (0, 0, 0),
        (0 + 1e-4, 0, 0),
        (0 + 1e-12, 0, 0),
        (0, 0, 1),
        (0, 0, 1 + 1e-4),
        (0, 0, 1 + 1e-12),
    ), dtype=tf.float32)

    with tf.GradientTape() as tape:
        normal = tf.Variable(normal)
        normal = mathutil.safe_l2_normalize(normal, axis=1)

        z = tf.convert_to_tensor((0, 0, 1), dtype=tf.float32) + eps
        z = tf.tile(z[None, :], (normal.shape[0], 1))

        t = tf.linalg.cross(normal, z)
        t = mathutil.safe_l2_normalize(t, axis=1)

        b = tf.linalg.cross(normal, t)
        b = mathutil.safe_l2_normalize(b, axis=1)
        b = mathutil.safe_l2_normalize(b, axis=1)
        b = mathutil.safe_l2_normalize(b, axis=1)
        b = mathutil.safe_l2_normalize(b, axis=1)

        rot = tf.stack((t, b, normal), axis=1)

    g = tape.gradient(rot, normal)
    max_mag = tf.reduce_max(tf.math.abs(g))
    print(max_mag)
    from IPython import embed; embed()


def l2_normalize():
    x = tf.Variable(tf.convert_to_tensor(np.array([
        [1, 1, 1],  # Grad OK
        [0, 0, 0],  # Grad NaN
        [1e-16, 1e-16, 1e-16],  # Grad OK
        [1e-19, 1e-19, 1e-19] #Grad Inf
    ], dtype=np.float32)))

    with tf.GradientTape() as tape1:
        y1 = mathutil.safe_l2_normalize(
            mathutil.safe_l2_normalize(
                mathutil.safe_l2_normalize(
                    x, axis=1),
                axis=1),
            axis=1)

    with tf.GradientTape() as tape2:
        y2 = tf.linalg.l2_normalize(
            tf.linalg.l2_normalize(
                tf.linalg.l2_normalize(
                    x, axis=1),
                axis=1),
            axis=1)

    g1 = tape1.gradient(y1, x)
    g2 = tape2.gradient(y2, x)


def main(_):
    microfacet()


if __name__ == '__main__':
    app.run(main)
