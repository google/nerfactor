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

import numpy as np
import tensorflow as tf

from nerfactor.util import math as mathutil


class Microfacet:
    """As described in:
        Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
    """
    def __init__(self, default_rough=0.3, lambert_only=False, f0=0.91):
        self.default_rough = default_rough
        self.lambert_only = lambert_only
        self.f0 = f0

    def __call__(self, pts2l, pts2c, normal, albedo=None, rough=None):
        """All in the world coordinates.

        Too low roughness is OK in the forward pass, but may be numerically
        unstable in the backward pass

        pts2l: NxLx3
        pts2c: Nx3
        normal: Nx3
        albedo: Nx3
        rough: Nx1
        """
        if albedo is None:
            albedo = tf.ones((tf.shape(pts2c)[0], 3), dtype=tf.float32)
        if rough is None:
            rough = self.default_rough * tf.ones(
                (tf.shape(pts2c)[0], 1), dtype=tf.float32)
        # Normalize directions and normals
        pts2l = mathutil.safe_l2_normalize(pts2l, axis=2)
        pts2c = mathutil.safe_l2_normalize(pts2c, axis=1)
        normal = mathutil.safe_l2_normalize(normal, axis=1)
        # Glossy
        h = pts2l + pts2c[:, None, :] # NxLx3
        h = mathutil.safe_l2_normalize(h, axis=2)
        f = self._get_f(pts2l, h) # NxL
        alpha = rough ** 2
        d = self._get_d(h, normal, alpha=alpha) # NxL
        g = self._get_g(pts2c, h, normal, alpha=alpha) # NxL
        l_dot_n = tf.einsum('ijk,ik->ij', pts2l, normal)
        v_dot_n = tf.einsum('ij,ij->i', pts2c, normal)
        denom = 4 * tf.abs(l_dot_n) * tf.abs(v_dot_n)[:, None]
        microfacet = tf.math.divide_no_nan(f * g * d, denom) # NxL
        brdf_glossy = tf.tile(microfacet[:, :, None], (1, 1, 3)) # NxLx3
        # Diffuse
        lambert = albedo / np.pi # Nx3
        brdf_diffuse = tf.broadcast_to(
            lambert[:, None, :], tf.shape(brdf_glossy)) # NxLx3
        # Mix two shaders
        if self.lambert_only:
            brdf = brdf_diffuse
        else:
            brdf = brdf_glossy + brdf_diffuse # TODO: energy conservation?
        return brdf # NxLx3

    @staticmethod
    def _get_g(v, m, n, alpha=0.1):
        """Geometric function (GGX).
        """
        cos_theta_v = tf.einsum('ij,ij->i', n, v)
        cos_theta = tf.einsum('ijk,ik->ij', m, v)
        denom = cos_theta_v[:, None]
        div = tf.math.divide_no_nan(cos_theta, denom)
        chi = tf.where(div > 0, 1., 0.)
        cos_theta_v_sq = tf.square(cos_theta_v)
        cos_theta_v_sq = tf.clip_by_value(cos_theta_v_sq, 0., 1.)
        denom = cos_theta_v_sq
        tan_theta_v_sq = tf.math.divide_no_nan(1 - cos_theta_v_sq, denom)
        tan_theta_v_sq = tf.clip_by_value(tan_theta_v_sq, 0., np.inf)
        denom = 1 + tf.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
        g = tf.math.divide_no_nan(chi * 2, denom)
        return g # (n_pts, n_lights)

    @staticmethod
    def _get_d(m, n, alpha=0.1):
        """Microfacet distribution (GGX).
        """
        cos_theta_m = tf.einsum('ijk,ik->ij', m, n)
        chi = tf.where(cos_theta_m > 0, 1., 0.)
        cos_theta_m_sq = tf.square(cos_theta_m)
        denom = cos_theta_m_sq
        tan_theta_m_sq = tf.math.divide_no_nan(1 - cos_theta_m_sq, denom)
        denom = np.pi * tf.square(cos_theta_m_sq) * tf.square(
            alpha ** 2 + tan_theta_m_sq)
        d = tf.math.divide_no_nan(alpha ** 2 * chi, denom)
        return d # (n_pts, n_lights)

    def _get_f(self, l, m):
        """Fresnel (Schlick's approximation).
        """
        cos_theta = tf.einsum('ijk,ijk->ij', l, m)
        f = self.f0 + (1 - self.f0) * (1 - cos_theta) ** 5
        return f # (n_pts, n_lights)
