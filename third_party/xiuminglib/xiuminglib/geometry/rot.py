import math
import numpy as np

from ..linalg import normalize

from ..log import get_logger
logger = get_logger()


def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        logger.warning((
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians"))


def axis_angle_to_rot_mat(axis, theta):
    r"""Gets rotation matrix that rotates points around an arbitrary axis by any
    angle.

    Rotating around the :math:`x`/:math:`y`/:math:`z` axis are special cases of
    this, where you simply specify the axis to be one of those axes.

    Args:
        axis (array_like): 3-vector that specifies the end point of the
            rotation axis (start point is the origin). This will be normalized
            to be unit-length.
        theta (float): Angle in radians, prescribed by the right-hand rule, so
            a negative value means flipping the rotation axis.

    Returns:
        numpy.ndarray: :math:`3\times 3` rotation matrix, to be pre-multiplied
        with the vector to rotate.
    """
    # NOTE: not tested thoroughly. Use with caution!
    axis = np.array(axis)

    ux, uy, uz = normalize(axis)
    cos = np.cos(theta)
    sin = np.sin(theta)

    r11 = cos + (ux ** 2) * (1 - cos)
    r12 = ux * uy * (1 - cos) - uz * sin
    r13 = ux * uz * (1 - cos) + uy * sin
    r21 = uy * ux * (1 - cos) + uz * sin
    r22 = cos + (uy ** 2) * (1 - cos)
    r23 = uy * uz * (1 - cos) - ux * sin
    r31 = uz * ux * (1 - cos) - uy * sin
    r32 = uz * uy * (1 - cos) + ux * sin
    r33 = cos + (uz ** 2) * (1 - cos)

    rmat = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return rmat


def is_rot_mat(mat, tol=1e-6):
    r"""Checks if a matrix is a valid rotation matrix.

    Args:
        mat (numpy.ndarray): A :math:`3\times 3` matrix.
        tol (float, optional): Tolerance for checking if all close.

    Returns:
        bool: Whether this is a valid rotation matrix.
    """
    mat_t = np.transpose(mat)
    should_be_identity = np.dot(mat_t, mat)
    identity = np.identity(3, dtype=mat.dtype)
    return np.allclose(identity, should_be_identity, atol=tol)


def rot_mat_to_euler_angles(rot_mat, tol=1e-6):
    r"""Converts a rotation matrix into Euler angles (rotation angles around
    the :math:`x`, :math:`y`, and :math:`z` axes).

    Args:
        rot_mat (numpy.ndarray): :math:`3\times 3` rotation matrix.
        tol (float, optional): Tolerance for checking singularity.

    Returns:
        numpy.ndarray: Euler angles in radians.
    """
    assert(is_rot_mat(rot_mat)), "Input matrix is not a valid rotation matrix"
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < tol
    if singular:
        x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = 0
    else:
        x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
    return np.array([x, y, z])
