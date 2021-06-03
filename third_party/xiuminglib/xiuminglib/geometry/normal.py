import numpy as np

from ..linalg import normalize as normalize_vec


def normalize(normal_map, norm_thres=0.5):
    """Normalizes the normal vector at each pixel of the normal map.

    Args:
        normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
        norm_thres (float, optional): Normalize only vectors with a norm
            greater than this; helpful to avoid errors at the boundary or
            in the background.

    Returns:
        numpy.ndarray: Normalized normal map.
    """
    norm = np.linalg.norm(normal_map, axis=-1)
    valid = norm > norm_thres
    normal_map[valid] = normalize_vec(normal_map[valid], axis=1)
    return normal_map


def transform_space(normal_map, rotmat):
    """Transforms the normal vectors from one space to another.

    Args:
        normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
        rotmat (numpy.ndarray or mathutils.Matrix): 3-by-3 rotation
            matrix, which is left-multiplied to the vectors.

    Returns:
        numpy.ndarray: Transformed normal map.
    """
    rotmat = np.array(rotmat)
    orig_shape = normal_map.shape
    normal = normal_map.reshape(-1, 3).T # 3-by-N

    normal_trans = rotmat.dot(normal)

    normal_map_trans = normal_trans.T.reshape(orig_shape)
    return normal_map_trans


def gen_world2local(normal):
    """Generates rotation matrices that transform world normals to local
    :math:`+z`, world tangents to local :math:`+x`, and world binormals to
    local :math:`+y`.

    Args:
        normal (numpy.ndarray): any size-by-3 array of normal vectors.

    Returns:
        numpy.ndarray: Any size-by-3-by-3 world-to-local rotation matrices,
        which should be left-multiplied to world coordinates.
    """
    last_dim_i = normal.ndim - 1

    z = np.array((0, 0, 1), dtype=float)

    # Tangents
    t = np.cross(normal, z)
    if (t == 0).all(axis=-1).any():
        raise ValueError((
            "Found (0, 0, 0) tangents! Possible reasons: normal colinear with "
            "(0, 0, 1); normal is (0, 0, 0)"))
    t = normalize_vec(t, axis=last_dim_i)

    # Binormals
    # No need to normalize because normals and tangents are orthonormal
    b = np.cross(normal, t)

    # Rotation matrices
    rot = np.stack((t, b, normal), axis=last_dim_i)
    # So that at each location, we have a 3x3 matrix whose ROWS, from top to
    # bottom, are world tangents, binormals, and normals

    return rot
