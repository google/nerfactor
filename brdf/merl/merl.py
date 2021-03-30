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

from os.path import basename
import numpy as np
from scipy.spatial import cKDTree

from third_party.xiuminglib import xiuminglib as xm
from third_party.nielsen2015on import merlFunctions as merl, \
    coordinateFunctions as coord


class MERL:
    def __init__(self, path=None):
        if path is None: # Lambertian
            cube_rgb = np.ones(coord.BRDFSHAPE, dtype=float)
            cube_rgb = np.tile(cube_rgb[:, :, :, None], (1, 1, 1, 3))
            name = 'lambertian'
        else:
            cube_rgb = merl.readMERLBRDF(path) # (phi_d, theta_h, theta_d, ch)
            name = self.parse_name(path)
        self._cube_rgb = cube_rgb
        self.name = name
        self.cube_rusink = self._get_merl_rusink(flat=False)
        self.flat_rusink = self._get_merl_rusink(flat=True)
        self.kdtree = None

    @property
    def cube_rgb(self):
        return self._cube_rgb

    @cube_rgb.setter
    def cube_rgb(self, x):
        correct_shape = self._cube_rgb.shape
        assert x.shape == correct_shape, \
            "Reflectance must be stored in a cube of shape %s" % correct_shape
        self._cube_rgb = x

    @property
    def flat_rgb(self):
        flat_rgb = np.reshape(self.cube_rgb, (-1, 3))
        return flat_rgb

    @property
    def tbl(self, keep_invalid=False):
        rusink_rgb = np.hstack((self.flat_rusink, self.flat_rgb))
        if not keep_invalid:
            # Discard invalid entries (-1's)
            valid = (rusink_rgb[:, 3:] > 0).all(axis=1)
            rusink_rgb = rusink_rgb[valid, :]
        return rusink_rgb

    @staticmethod
    def parse_name(path):
        return basename(path)[:-len('.binary')]

    @staticmethod
    def _get_merl_rusink(flat=False):
        ind = np.indices(coord.BRDFSHAPE) # 3x180x90x90
        ind_flat = np.reshape(ind, (3, -1)).T
        rusink_flat = coord.MERLToRusink(ind_flat) # (phi_d, theta_h, theta_d)
        if flat:
            return rusink_flat
        rusink_cube = np.reshape(rusink_flat, coord.BRDFSHAPE + (3,))
        return rusink_cube

    def get_characterstic_slice(self):
        """Characteristic slice (phi_d = 90):
            ^ theta_d
            |
            |
            +-------> theta_h
        """
        phi_i = self.cube_rgb.shape[0] // 2
        cslice = self.cube_rgb[phi_i, :, :]
        cslice = np.rot90(cslice, axes=(0, 1))
        return cslice

    def get_characterstic_slice_rusink(self):
        """Rusink. coordinates for the characteristic slice (see above).
        """
        phi_i = self.cube_rusink.shape[0] // 2
        rusink = self.cube_rusink[phi_i, :, :, :]
        rusink = np.rot90(rusink, axes=(0, 1))
        return rusink

    @staticmethod
    def characteristic_slice_as_img(cslice, clip_percentile=80):
        maxv = np.percentile(cslice, clip_percentile)
        cslice_0to1 = np.clip(cslice, 0, maxv) / maxv
        cslice_uint = xm.img.denormalize_float(cslice_0to1)
        cslice_img = xm.img.gamma_correct(cslice_uint)
        return cslice_img

    @staticmethod
    def dir2rusink(ldir, vdir):
        """Inputs must be in local frames, of shapes:
            - `ldir`: HxWxLx3; and
            - `vdir`: HxWx3.
        """
        ldir_flat = np.reshape(ldir, (-1, 3))
        vdir_rep = np.tile(vdir[:, :, None, :], (1, 1, ldir.shape[2], 1))
        vdir_flat = np.reshape(vdir_rep, (-1, 3))
        rusink = coord.DirectionsToRusink(vdir_flat, ldir_flat)
        rusink = np.reshape(rusink, ldir.shape)
        return rusink # (phi_d, theta_h, theta_d)

    def query(self, qrusink):
        """Nearest neighbor lookup, sped up by k-D tree.

        `qrusink`: Nx3.
        """
        # Use cached tree or build one
        if self.kdtree is None:
            self.kdtree = cKDTree(self.tbl[:, :3])
        _, ind = self.kdtree.query(qrusink)
        rgb = self.tbl[ind, 3:]
        return rgb
