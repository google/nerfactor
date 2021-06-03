# pylint: disable=too-many-public-methods

import json
import numpy as np

from .geometry.proj import to_homo, from_homo
from .geometry.rot import is_rot_mat, rot_mat_to_euler_angles
from .linalg import normalize


GLCAM_TO_CVCAM = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
CVCAM_TO_GLCAM = np.linalg.inv(GLCAM_TO_CVCAM)


class PerspCam:
    r"""Perspective camera in 35mm format.

    This is not an OpenGL/Blender camera (where :math:`+x` points right,
    :math:`+y` up, and :math:`-z` into the viewing direction), but rather a
    "CV camera" (where :math:`+x` points right, :math:`+y` down, and :math:`+z`
    into the viewing direction). See more in :attr:`~ext_mat`.

    Because we mostly consider just the camera and the object, we assume the
    object coordinate system (the "local system" in Blender) aligns with (and
    hence, is the same as) the world coordinate system (the "global system" in
    Blender).

    Note:
        - Sensor width of the 35mm format is actually 36mm.
        - This class assumes unit pixel aspect ratio (i.e., :math:`f_x = f_y`)
          and no skewing between the sensor plane and optical axis.
        - The active sensor size may be smaller than ``sensor_w`` and
          ``sensor_h``, depending on ``im_res``. See :attr:`~sensor_w_active`
          and :attr:`~sensor_h_active`.
        - ``aov``, ``sensor_h``, and ``sensor_w`` are hardware properties,
          having nothing to do with ``im_res``.
    """
    def __init__(
            self, name='cam', f_pix=533.33, im_res=(256, 256),
            loc=(1, 1, 1), lookat=(0, 0, 0), up=(0, 1, 0)):
        """
        Args:
            name (str, optional): Camera name.
            f_pix (float, optional): Focal length in pixel.
            im_res (array_like, optional): Image height and width in pixels.
            loc (array_like, optional): Camera location in object space.
            lookat (array_like, optional): Where the camera points to in
                object space, so default :math:`(0, 0, 0)` is the object center.
            up (array_like, optional): Vector in object space that, when
                projected, points upward in image.
        """
        self._name = str(name)
        self._f_pix = float(f_pix)
        self._im_h = int(im_res[0])
        self._im_w = int(im_res[1])
        self._loc = np.array(loc)
        self._lookat = np.array(lookat)
        self._up = np.array(up)

    @property
    def name(self):
        """str: Camera name."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def f_pix(self):
        """float: Focal length in pixels."""
        return self._f_pix

    @f_pix.setter
    def f_pix(self, value):
        self._f_pix = float(value)

    @property
    def im_h(self):
        """int: Image height.
        """
        return self._im_h

    @im_h.setter
    def im_h(self, value):
        self._im_h = safe_cast_to_int(value)

    @property
    def im_w(self):
        """int: Image width.
        """
        return self._im_w

    @im_w.setter
    def im_w(self, value):
        self._im_w = safe_cast_to_int(value)

    @property
    def loc(self):
        """numpy.ndarray: Camera location in object space."""
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = np.array(value)

    @property
    def lookat(self):
        """numpy.ndarray: Where in object space the camera points to."""
        return self._lookat

    @lookat.setter
    def lookat(self, value):
        self._lookat = np.array(value)

    @property
    def up(self):
        """numpy.ndarray: Up vector, the vector in object space that, when
        projected, points upward on image plane.
        """
        return self._up

    @up.setter
    def up(self, value):
        self._up = np.array(value)

    @property
    def aov(self):
        """numpy.ndarray: Vertical and horizontal angles of view in degrees."""
        alpha_v = 2 * np.arctan(self.sensor_h / (2 * self.f_mm))
        alpha_h = 2 * np.arctan(self.sensor_w / (2 * self.f_mm))
        alpha = np.array([alpha_v, alpha_h])
        return alpha / np.pi * 180

    @property
    def sensor_w(self):
        """float: Sensor's physical width (fixed at 36mm)."""
        return 36 # mm

    @property
    def sensor_h(self):
        """float: Sensor's physical height (fixed at 24mm)."""
        return 24 # mm

    @property
    def sensor_fit_horizontal(self):
        """bool: Whether field of view angle fits along the horizontal or
        vertical direction.
        """
        if self.sensor_h / self.im_h < self.sensor_w / self.im_w:
            return False
        return True

    @property
    def mm_per_pix(self):
        """float: Millimeter per pixel."""
        if self.sensor_fit_horizontal:
            return self.sensor_w / self.im_w
        return self.sensor_h / self.im_h

    @property
    def sensor_w_active(self):
        """float: Actual sensor width (mm) in use (resolution-dependent)."""
        return self.im_w * self.mm_per_pix

    @property
    def sensor_h_active(self):
        """float: Actual sensor height (mm) in use (resolution-dependent)."""
        return self.im_h * self.mm_per_pix

    @property
    def f_mm(self):
        """float: 35mm format-equivalent focal length in mm."""
        return self.mm_per_pix * self.f_pix

    @f_mm.setter
    def f_mm(self, value):
        self._f_pix = float(value) / self.mm_per_pix

    @property
    def int_mat(self):
        r"""numpy.ndarray: :math:`3\times 3` intrinsics matrix."""
        return np.array([
            [self.f_pix, 0, self.im_w / 2],
            [0, self.f_pix, self.im_h / 2],
            [0, 0, 1]])

    @int_mat.setter
    def int_mat(self, mat):
        mat = np.array(mat)
        # Assert matrix structure
        assert mat.shape == (3, 3), "Intrinsics matrix is not 3x3"
        assert mat[1, 0] == 0, "`intrinsics[1, 0]` is not 0"
        skew = mat[0, 1]
        assert skew == 0, f"Skew ({skew}) is not 0"
        assert all(mat[2, :] == [0, 0, 1]), "Last row is not [0, 0, 1]"
        f_pix = mat[0, 0]
        assert f_pix == mat[1, 1], "X and Y focal lengths are different"
        # Set relevant properties
        self.f_pix = f_pix
        self.im_w = mat[0, 2] * 2
        self.im_h = mat[1, 2] * 2

    @property
    def ext_mat(self):
        r"""numpy.ndarray: :math:`3\times 4` object-to-camera extrinsics matrix,
        i.e., rotation and translation that transform a point from object space
        to camera space.

        Two coordinate systems involved: object space "obj" and camera space
        following the computer vision convention "cv", where :math:`+x`
        horizontally points right (to align with pixel coordinates), :math:`+y`
        vertically points down, and :math:`+z` is the look-at direction
        (because right-handed).
        """
        # cv axes expressed in obj space
        cvz_obj = self.lookat - self.loc
        assert np.linalg.norm(cvz_obj) > 0, \
            "Camera location and look-at coincide"
        cvx_obj = np.cross(cvz_obj, self.up)
        cvy_obj = np.cross(cvz_obj, cvx_obj)
        # Normalize
        cvz_obj = normalize(cvz_obj)
        cvx_obj = normalize(cvx_obj)
        cvy_obj = normalize(cvy_obj)
        # Compute rotation from obj to cv: R
        # cvx_obj gives first column of R, cvy_obj second, and cvz_obj third
        rot_obj2cv = np.vstack((cvx_obj, cvy_obj, cvz_obj)).T
        # Extrinsics
        obj2cv = rot_obj2cv.dot( # translate first and then rotate
            np.array([
                [1, 0, 0, -self.loc[0]],
                [0, 1, 0, -self.loc[1]],
                [0, 0, 1, -self.loc[2]]]))
        return obj2cv

    @ext_mat.setter
    def ext_mat(self, o2c):
        o2c = np.array(o2c)
        # Assert matrix structure
        assert o2c.shape == (3, 4), "This setter accepts only 3x4 extrinsics"
        r_o2c = o2c[:, :3]
        assert is_rot_mat(r_o2c), \
            "The R part of object-to-camera is not a valid rotation matrix"
        # Compute camera location in object space
        t_o2c = o2c[:, 3]
        self.loc = -np.linalg.inv(r_o2c).dot(t_o2c)
        # Compute look-at in object space
        cvz_obj = r_o2c[:, 2]
        self.lookat = cvz_obj + self.loc
        # Compute up in object space
        cvx_obj = r_o2c[:, 0]
        self.up = np.cross(cvx_obj, cvz_obj) / cvz_obj.dot(cvz_obj)
        # TODO: Why does the following not work?
        '''
        # Camera to object space
        c2o = np.linalg.inv(o2c)
        # Camera location in object space is the origin of camera space
        loc = c2o.dot(to_homo([0, 0, 0]))
        self.loc = from_homo(loc)
        # Look-at in object space is any point on +z in camera space
        lookat = c2o.dot(to_homo([0, 0, 1]))
        self.lookat = from_homo(lookat)
        # Up vector in object space is -y in camera space
        up = c2o.dot(to_homo([0, -1, 0]))
        self.up = from_homo(up)
        '''

    @property
    def ext_mat_4x4(self):
        r"""numpy.ndarray: Padding :math:`[0, 0, 0, 1]` to bottom of the
        :math:`3\times 4` extrinsics matrix to make it invertible.
        """
        return np.vstack((self.ext_mat, [0, 0, 0, 1]))

    @ext_mat_4x4.setter
    def ext_mat_4x4(self, o2c):
        o2c = np.array(o2c)
        # Assert matrix structure
        assert o2c.shape == (4, 4), "This setter accepts only 4x4 extrinsics"
        assert all(o2c[3, :] == [0, 0, 0, 1]), \
            "Last row of 4x4 extrinsics must be [0, 0, 0, 1]"
        # Call the 3x4 setter
        self.ext_mat = o2c[:3, :]

    @property
    def proj_mat(self):
        r"""numpy.ndarray: :math:`3\times 4` projection matrix, derived from
        intrinsics and extrinsics.
        """
        return self.int_mat.dot(self.ext_mat)

    @property
    def blender_rot_euler(self):
        """numpy.ndarray: Euler rotations in degrees."""
        c2o = self.get_cam2obj(cam_type='blender')
        rot_mat = c2o[:3, :3]
        euler_angles = rot_mat_to_euler_angles(rot_mat)
        return euler_angles / np.pi * 180

    def to_dict(self, app=None):
        """Converts this camera to a dictionary of its properties.

        Args:
            app (str, optional): For what application are we converting?
                Accepted are ``None`` and ``'blender'``.

        Returns:
            dict: This camera as a dictionary.
        """
        if isinstance(app, str):
            app = app.lower()
        if app is None:
            prop_dict = {
                'name': self.name, 'f_mm': self.f_mm, 'f_pix': self.f_pix,
                'sensor_fit_horizontal': self.sensor_fit_horizontal,
                'sensor_w': self.sensor_w,
                'sensor_w_active': self.sensor_w_active,
                'sensor_h': self.sensor_h,
                'sensor_h_active': self.sensor_h_active,
                'mm_per_pix': self.mm_per_pix,
                'im_h': self.im_h, 'im_w': self.im_w,
                'loc': self.loc, 'lookat': self.lookat, 'up': self.up,
                'aov': self.aov, 'int_mat': self.int_mat,
                'ext_mat': self.ext_mat, 'proj_mat': self.proj_mat}
        elif app == 'blender':
            prop_dict = {
                'name': self.name, 'f_mm': self.f_mm,
                'im_h': self.im_h, 'im_w': self.im_w,
                'sensor_fit_horizontal': self.sensor_fit_horizontal,
                'sensor_h': self.sensor_h, 'sensor_w': self.sensor_w,
                'loc': self.loc, 'rot_euler_deg': self.blender_rot_euler}
        else:
            raise NotImplementedError(app)
        return prop_dict

    def __str__(self):
        prop_dict = self.to_dict()
        prop_dict_serializable = {}
        for k, v in prop_dict.items():
            if isinstance(v, np.ndarray):
                prop_dict_serializable[k] = v.tolist()
            else:
                prop_dict_serializable[k] = v
        prop_str = json.dumps(prop_dict_serializable, indent=4)
        return prop_str

    def get_obj2cam(self, cam_type='cv', square=False):
        r"""Gets the object-to-camera transformation matrix.

        Args:
            cam_type (str, optional): Accepted are ``'cv'``/``'opencv'`` and
                ``'opengl'``/``'blender'``.
            square (bool, optional): If true, the last row of
                :math:`[0, 0, 0, 1]` is kept, which makes the matrix invertible.

        Returns:
            numpy.ndarray: :math:`3\times 4` or :math:`4\times 4`
            object-to-camera transformation matrix.
        """
        cam_type = cam_type.lower()
        if cam_type in ('cv', 'opencv'):
            obj2cam = self.ext_mat
        elif cam_type in ('gl', 'opengl', 'blender'):
            # Additional 180-degree rotation around x-axis
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            obj2cam = rot.dot(self.ext_mat)
        else:
            raise NotImplementedError(f"Camera type: {cam_type}")
        if square:
            obj2cam_4x4 = np.vstack((obj2cam, [0, 0, 0, 1]))
            return obj2cam_4x4
        return obj2cam

    def get_cam2obj(self, cam_type='cv', square=False):
        """Inverse of :func:`get_obj2cam`.

        One example use: calling this with ``cam_type='blender'`` gives
        Blender's ``cam.matrix_world``.
        """
        obj2cam_4x4 = self.get_obj2cam(cam_type=cam_type, square=True)
        cam2obj_4x4 = np.linalg.inv(obj2cam_4x4)
        if square:
            return cam2obj_4x4
        return cam2obj_4x4[:3, :]

    def set_from_mitsuba(self, xml_path):
        """Sets camera according to a Mitsuba XML file.

        Args:
            xml_path (str): Path to the XML file.
        """
        from xml.etree.ElementTree import parse

        tree = parse(xml_path)
        # Focal length
        f_tag = tree.find('./sensor/string[@name="focalLength"]')
        if f_tag is None:
            self.f_mm = 50. # Mitsuba default
        else:
            f_str = f_tag.attrib['value']
            if f_str[-2:] == 'mm':
                self.f_mm = float(f_str[:-2])
            else:
                raise NotImplementedError(f_str)
        # Extrinsics
        cam_transform = tree.find('./sensor/transform/lookAt').attrib
        self.loc = np.fromstring(cam_transform['origin'], sep=',')
        self.lookat = np.fromstring(cam_transform['target'], sep=',')
        self.up = np.fromstring(cam_transform['up'], sep=',')
        # Resolution
        self.im_h = int(
            tree.find('./sensor/film/integer[@name="height"]').attrib['value'])
        self.im_w = int(
            tree.find('./sensor/film/integer[@name="width"]').attrib['value'])

    def proj(self, pts, space='object'):
        r"""Projects 3D points to 2D.

        Args:
            pts (array_like): 3D point(s) of shape :math:`N\times 3` or
                :math:`3\times N`, or of length 3.
            space (str, optional): In which space these points are specified:
                ``'object'`` or ``'camera'``.

        Returns:
            array_like: Vertical and horizontal coordinates of the projections,
            following:

            .. code-block:: none

                +-----------> dim1
                |
                |
                |
                v dim0
        """
        pts = np.array(pts)
        if pts.shape == (3,):
            pts = pts.reshape((3, 1))
        elif pts.shape[1] == 3:
            pts = pts.T
        assert space in ('object', 'camera'), "Unrecognized space"
        pts_homo = to_homo(pts) # 3xN to 4xN
        if space == 'object':
            proj_mat = self.proj_mat
        else:
            ext_mat = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj_mat = self.int_mat.dot(ext_mat)
        # Project
        hvs_homo = proj_mat.dot(pts_homo)
        # 3xN: dim0 is horizontal, and dim1 is vertical
        hvs = from_homo(hvs_homo) # 3xN to 2xN
        vhs = np.vstack((hvs[1, :], hvs[0, :])).T
        if vhs.shape[0] == 1:
            # Single point
            vhs = vhs[0, :]
        return vhs

    def backproj(
            self, depth, fg_mask=None, bg_fill=0., depth_type='plane',
            space='object'):
        """Backprojects a depth map to 3D points.

        Resolution of the depth map may be different from :attr:`im_h` and
        :attr:`im_w`: :attr:`im_h` and :attr:`im_w` decide the image coordinate
        bounds, and the depth resolution decides number of steps.

        Args:
            depth (numpy.ndarray): Depth map.
            fg_mask (numpy.ndarray, optional): Backproject only pixels falling
                inside this foreground mask. Its values should be logical.
            bg_fill (flaot, optional): Filler value for background region.
            depth_type (str, optional): Plane or ray depth.
            space (str, optional): In which space the backprojected points are
                specified: ``'object'`` or ``'camera'``.

        Returns:
            numpy.ndarray: :math:`xyz` map.
        """
        if fg_mask is None:
            fg_mask = np.ones(depth.shape, dtype=bool)
        assert depth_type in ('ray', 'plane'), "Unrecognized depth type"
        assert space in ('object', 'camera'), "Unrecognized space"
        # Generate 2D coordinates
        v_is, h_is = np.where(fg_mask)
        hs = (h_is + 0.5) / fg_mask.shape[1] * self.im_w
        vs = (v_is + 0.5) / fg_mask.shape[0] * self.im_h
        h_c, v_c = self.im_w / 2, self.im_h / 2
        zs = depth[fg_mask]
        if depth_type == 'ray':
            d2 = np.power(vs - v_c, 2) + np.power(hs - h_c, 2)
            # Similar triangles
            zs_plane = np.multiply(
                zs, self.f_pix / np.sqrt(self.f_pix ** 2 + d2))
            zs = zs_plane
        # Backproject to camera space
        xs = np.multiply(zs, hs - h_c) / self.f_pix
        ys = np.multiply(zs, vs - v_c) / self.f_pix
        pts = np.vstack((xs, ys, zs))
        if space == 'object':
            # Need to further transform to object space
            o2c = self.ext_mat_4x4
            c2o = np.linalg.inv(o2c)
            pts_o = c2o.dot(to_homo(pts))
            pts = from_homo(pts_o, axis=0)
        pts = pts.T # (n_fg_pts, 3)
        # Put them back into a buffer
        xyz = bg_fill * np.ones(depth.shape + (3,), dtype=float)
        xyz[np.dstack([fg_mask] * 3)] = pts.ravel()
        return xyz

    def gen_rays(self, spp=1):
        r"""Generates ray directions in object space, with the ray origin being
        the camera location.

        Args:
            spp (int, optional): Samples (or number of rays) per pixel. Must be
                a perfect square :math:`S^2` due to uniform, deterministic
                supersampling.

        Returns:
            numpy.ndarray: An :math:`H\times W\times S^2\times 3` array of ray
            directions.
        """
        sps = np.sqrt(spp)
        if sps.is_integer():
            sps = int(sps)
        else:
            raise ValueError(
                f"Samples per pixel ({spp}) is not a perfect square")
        # Supersample according to samples per side
        h, w = self.im_h * sps, self.im_w * sps
        depth = np.ones((h, w), dtype=float)
        # Backproject a uniform plane depth map to a wall in 3D
        xyzs = self.backproj(depth) # (HS)x(WS)x3
        # Compute ray directions
        ray_dirs_ss = normalize(xyzs - self.loc, axis=2)
        # Put samples in each pixel bucket
        ray_dirs = []
        for i in range(sps):
            for j in range(sps):
                ray_dirs.append(ray_dirs_ss[i::sps, j::sps, :])
        ray_dirs = np.stack(ray_dirs, axis=2)
        return ray_dirs # HxWx(S^2)x3

    def resize(self, new_h=None, new_w=None):
        """Updates the camera intrinsics according to the new size.

        Args:
            new_h (int, optional): Target height. If ``None``, will be
                calculated according to the target width, assuming the same
                aspect ratio.
            new_w (int, optional): Target width. If ``None``, will be calculated
                according to the target height, assuming the same aspect ratio.
        """
        if new_h is not None and new_w is not None:
            assert int(self.im_h / self.im_w * new_w) == new_h, \
                "Aspect ratio change violates the `f_x == f_y` assumption"
        elif new_h is None and new_w is not None:
            new_h = int(self.im_h / self.im_w * new_w)
        elif new_h is not None and new_w is None:
            new_w = int(self.im_w / self.im_h * new_h)
        else:
            raise ValueError(
                "At least one of new height or width must be given")
        # Update relevant properties
        self.f_pix = new_h / float(self.im_h) * self.f_pix
        self.im_h = new_h
        self.im_w = new_w


def safe_cast_to_int(x):
    """Casts a string or float to integer only when safe.

    Args:
        x (str or float): Input to be cast to integer.

    Returns:
        int: Integer version of the input.
    """
    int_x = int(x)
    if np.issubdtype(type(x), np.floating):
        assert int_x == x, \
            f"Can't safely cast a non-integer value ({x}) to integer"
    return int_x
