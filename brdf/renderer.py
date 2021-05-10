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

# pylint: disable=invalid-unary-operand-type,unsupported-assignment-operation

from os.path import join
import numpy as np

from third_party.xiuminglib import xiuminglib as xm


class SphereRenderer:
    """Renders a sphere of given BRDF under given environment map.

    Assumptions:
        1. Direct illumination only (no global illumination or multiple
           bounces);
        2. Fixed and uniform sampling of the environment map (as if you were
           approximating the map with a light stage of LEDs);
    """
    def __init__(
            self, envmap_path, out_dir, envmap_inten=1., envmap_h=None,
            ims=128, spp=1, debug=False):
        self.out_dir = out_dir
        self.ims = ims
        self.debug = debug
        self.sps = self._spp2sps(spp)
        # Scene
        self.cam, self.xyz, self.is_fg = self._gen_scene()
        self.normal = self._calc_normals()
        self.world2local = xm.geometry.normal.gen_world2local( # world-to-local
            self.normal) # matrices that transform +Z to local normals
        # Lighting
        light_vis = join(self.out_dir, 'debug', 'light.png') if debug else None
        envmap = load_light(
            envmap_path, envmap_inten=envmap_inten, envmap_h=envmap_h,
            vis_path=light_vis)
        self.lxyz, self.lareas = gen_light_xyz(*envmap.shape[:2])
        # Directions
        self.ldir = self.gen_light_dir(local=True)
        self.vdir = self.gen_view_dir(local=True)
        # Cosines
        self.lcos = self._calc_cosines()
        # Light visibility
        is_front_lit = self.lcos > 0
        is_fg_rep = np.tile(self.is_fg[:, :, None], (1, 1, self.ldir.shape[2]))
        lvis = np.logical_and(is_fg_rep, is_front_lit)
        self.lvis = lvis.astype(float)
        # Calculate each LED's contribution to each pixel
        self.lcontrib = self.calc_light_contrib(envmap)

    @staticmethod
    def _spp2sps(spp):
        """Computes samples per side from samples per pixel.
        """
        sps = np.sqrt(spp)
        assert sps == int(sps), "`spp` must be a square integer"
        return int(sps)

    def _gen_scene(self, sphere_radius=0.4, cam_dist=10.):
        """A sphere sits at the origin, and the camera sits on -Z, looking at
        origin with +Y as the up vector.
        """
        # Depth map
        assert cam_dist > sphere_radius
        c = np.array((.5, .5))
        sample_w = 1 / (self.sps + 1)
        x = np.linspace(
            sample_w, self.ims - sample_w, self.ims * self.sps, endpoint=True)
        x /= self.ims
        xx, yy = np.meshgrid(x, x)
        xy = np.dstack((xx, yy))
        dist = np.linalg.norm(
            xy - c[None, None, :], axis=2) # distance to map center
        is_fg = dist <= sphere_radius
        height = np.sqrt( # height along Z of a sphere
            np.where(is_fg, sphere_radius ** 2 - np.square(dist), 0))
        depth = cam_dist - height # [cam_dist - radius, cam_dist]
        # XYZ buffer by backprojecting with camera
        cam = xm.camera.PerspCam(
            im_res=(self.ims * self.sps, self.ims * self.sps),
            loc=(0, 0, -cam_dist))
        cam.f_mm = cam_dist * ( # so radius projected to desired sensor range
            cam.sensor_h_active / 2 / 0.5 * sphere_radius) / sphere_radius
        xyz = cam.backproj( # NOTE: their radii are close to,
            depth, fg_mask=is_fg) # but not exactly, the sphere radius
        if self.debug:
            plot = xm.vis.plot.Plot()
            plot.scatter3d(
                xyz[is_fg], views=[(30, 45), (30, 135)], equal_axes=True,
                outpath=join(self.out_dir, 'debug', 'xyz.png'))
        return cam, xyz, is_fg

    def _calc_normals(self, eps=1e-12):
        # Normal buffer
        normal = self.xyz - 0 # since sphere center is at origin
        normal += eps # to avoid problem (0, 0, 0) and (0, 0, +/-1)
        normal = xm.linalg.normalize(normal, axis=2)
        if self.debug:
            normal_vis = (normal + 1) / 2
            xm.io.img.write_arr(
                normal_vis, join(self.out_dir, 'debug', 'normal.png'))
        return normal

    def gen_view_dir(self, local=False):
        vdir = self.cam.loc[None, None, :] - self.xyz # (H, W, 3)
        if local:
            vdir = np.einsum('ijkl,ijl->ijk', self.world2local, vdir)
        vdir = xm.linalg.normalize(vdir, axis=2)
        return vdir

    def gen_light_dir(self, local=False):
        lxyz_flat = np.reshape(self.lxyz, (-1, 3))
        ldir = lxyz_flat[None, None, :, :] - \
            self.xyz[:, :, None, :] # (H, W, L, 3)
        if local:
            ldir = np.einsum('ijkl,ijnl->ijnk', self.world2local, ldir)
        ldir = xm.linalg.normalize(ldir, axis=3)
        return ldir

    def _calc_cosines(self):
        lcos = self.ldir @ np.array( # because light directions are in the local
            (0, 0, 1)) # space, normals are always +Z
        # Optionally, visualize cosines as a video
        if self.debug:
            alpha = self.is_fg.astype(float)
            lcos_pos = np.clip(lcos, 0, 1)
            frames = [lcos_pos[:, :, i] for i in range(lcos_pos.shape[2])]
            frames = [alpha * x for x in frames]
            frames = [np.dstack([x] * 3) for x in frames]
            frames = [(x * 255).astype(np.uint8) for x in frames]
            xm.vis.video.make_video(
                frames, method='video_api', outpath=join(
                    self.out_dir, 'debug', 'lcos.webm'))
        return lcos

    def calc_light_contrib(self, light):
        # Shape matching
        light = np.reshape(light, (-1, 3))
        light = np.tile(
            light[None, None, :, :],
            (self.ims * self.sps, self.ims * self.sps, 1, 1))
        lareas = np.reshape(self.lareas, (-1,))
        lareas = np.tile(
            lareas[None, None, :],
            (self.ims * self.sps, self.ims * self.sps, 1,))
        # NOTE: light visibility should include the "front lit" criterion, and
        # other occulusion
        lvis = np.tile(self.lvis[:, :, :, None], (1, 1, 1, 3))
        # Mask out non-visible lights
        light = lvis * light
        # Compute light contribution
        light_contrib = light * self.lcos[:, :, :, None] * lareas[:, :, :, None]
        return light_contrib

    def render(self, brdf, white_bg=True):
        """Input `brdf` should be of the same shape as `self.lcontrib`: HxWxLx3.
        """
        render = brdf * self.lcontrib
        render = np.sum(render, axis=2)
        # Composite on background
        is_fg_rgb = np.dstack([self.is_fg] * 3)
        render[~is_fg_rgb] = 1. if white_bg else 0.
        # Now what is the real resolution? Average pixel samples
        render_sum = np.zeros((self.ims, self.ims, 3), dtype=render.dtype)
        for i in range(self.sps):
            for j in range(self.sps):
                render_sum += render[i::self.sps, j::self.sps, :]
        render_avg = render_sum / (self.sps ** 2)
        return render_avg


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = xm.geometry.sph.sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas


def load_light(envmap_path, envmap_inten=1., envmap_h=None, vis_path=None):
    if envmap_path == 'white':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.ones((h, 2 * h, 3), dtype=float)

    elif envmap_path == 'point':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.zeros((h, 2 * h, 3), dtype=float)
        i = -envmap.shape[0] // 4
        j = -int(envmap.shape[1] * 7 / 8)
        d = 2
        envmap[(i - d):(i + d), (j - d):(j + d), :] = 1

    else:
        envmap = xm.io.exr.read(envmap_path)

    # Optionally resize
    if envmap_h is not None:
        envmap = xm.img.resize(envmap, new_h=envmap_h)

    # Scale by intensity
    envmap = envmap_inten * envmap

    # visualize the environment map in effect
    if vis_path is not None:
        xm.io.img.write_arr(envmap, vis_path, clip=True)

    return envmap
