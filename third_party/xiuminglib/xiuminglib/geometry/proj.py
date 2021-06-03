import numpy as np

from ..imprt import preset_import


def to_homo(pts):
    """Pads 2/3D points to homogeneous, by guessing which dimension to pad.

    Args:
        pts (array_like): Input array of 2D or 3D points.

    Returns:
        numpy.ndarray: Homogeneous coordinates of the input points.
    """
    pts = np.array(pts)

    if pts.ndim == 1:
        pts_homo = np.hstack((pts, 1))

    elif pts.ndim == 2:
        err_str = " (assumed to be # points) must be >3 to be not ambiguous"
        h, w = pts.shape
        if h > w: # tall
            assert h > 3, "Input has height (%d) > width (%d); the height" \
                % (h, w) + err_str
            pts_homo = np.hstack((pts, np.ones((h, 1))))
        elif h < w: # fat
            assert w > 3, "Input has width (%d) > height (%d); the width" \
                % (w, h) + err_str
            pts_homo = np.vstack((pts, np.ones((1, w))))
        else: # square
            raise ValueError(
                "Ambiguous square matrix that I can't guess how to pad")

    else:
        raise ValueError(pts.ndim)

    return pts_homo


def from_homo(pts, axis=None):
    """Converts from homogeneous to non-homogeneous coordinates.

    Args:
        pts (numpy.ndarray or mathutils.Vector): NumPy array of N-D point(s),
            or Blender vector of a single N-D point.
        axis (int, optional): The last slice of which dimension holds the
            :math:`w` values. Optional for 1D inputs.

    Returns:
        numpy.ndarray or mathutils.Vector: Non-homogeneous coordinates of the
        input point(s).
    """
    Vector = preset_import('Vector')

    if Vector is not None and isinstance(pts, Vector):
        if axis not in (None, 0):
            raise ValueError((
                "Axis must be either None (auto) or 0 for a Blender vector "
                "input"))
        pts_nonhomo = Vector(x / pts[-1] for x in pts[:-1])

    elif isinstance(pts, np.ndarray):
        if axis is None:
            if pts.ndim == 1:
                axis = 0
            else:
                raise ValueError((
                    "When pts has more than one dimension, axis must be "
                    "specified"))
        arr = np.take(pts, range(pts.shape[axis] - 1), axis=axis)
        w = np.take(pts, -1, axis=axis)
        pts_nonhomo = np.divide(arr, w) # by broadcasting

    else:
        raise TypeError(pts)

    return pts_nonhomo
