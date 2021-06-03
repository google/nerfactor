import numpy as np


def is_symmetric(mat, eps=None):
    """Checks if a matrix is symmetric.

    If the input is not even square, ``False`` is returned.

    Args:
        mat (numpy.ndarray): Input matrix.
        eps (float, optional): Numerical tolerance for equality. ``None``
            means ``np.finfo(mat.dtype).eps``.

    Returns:
        bool: Whether the input is symmetric.
    """
    if eps is None:
        eps = np.finfo(mat.dtype).eps

    assert mat.ndim == 2
    if mat.shape[0] != mat.shape[1]:
        return False

    return np.allclose(mat, mat.T, atol=eps)


def is_identity(mat, eps=None):
    """Checks if a matrix is an identity matrix.

    If the input is not even square, ``False`` is returned.

    Args:
        mat (numpy.ndarray): Input matrix.
        eps (float, optional): Numerical tolerance for equality. ``None``
            means ``np.finfo(mat.dtype).eps``.

    Returns:
        bool: Whether the input is an identity matrix.
    """
    if eps is None:
        eps = np.finfo(mat.dtype).eps

    assert mat.ndim == 2
    if mat.shape[0] != mat.shape[1]:
        return False

    return np.allclose(mat, np.eye(mat.shape[0]), atol=eps)


def angle_between(vec1, vec2, radian=True):
    r"""Computes the angle between two vectors.

    Args:
        vec1 (array_like): Vector 1.
        vec2
        radian (bool, optional): Whether to use radians.

    Returns:
        float: The angle :math:`\in [0,\pi]`.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cos = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    angle = np.arccos(np.clip(cos, -1, 1))
    if not radian:
        angle = angle / np.pi * 180
    return angle


def normalize(vecs, axis=0):
    """Normalizes vectors.

    Args:
        vecs (array_like): 1D array for a single vector, 2D array for multiple
            vectors, 3D array for an "image" of vectors, etc.
        axis (int, optional): Along which axis normalization is done.

    Returns:
        numpy.ndarray: Normalized vector(s) of the same shape as input.
    """
    vecs = np.array(vecs)

    if axis < 0:
        raise ValueError("Negative index not allowed for safety")
    n_dims = vecs.ndim
    if axis >= n_dims:
        raise ValueError((
            "Can't normalize along axis %d when you only have %d dimension(s)"
        ) % (axis, n_dims))

    norms = np.linalg.norm(vecs, axis=axis)
    if (norms == 0.).any():
        raise ValueError("Found zero-norm vectors")

    broadcastable = list(vecs.shape)
    broadcastable[axis] = 1
    vecs_normalized = np.divide(vecs, norms.reshape(broadcastable))

    return vecs_normalized


def project_onto(pts, basis):
    """Projects points onto a basis vector.

    Args:
        pts (array_like): 1D array for one vector; 2D N-by-M array for N
            M-D points.
        basis (array_like): 1D M-array specifying which basis vector to
            project to.

    Returns:
        numpy.ndarray: Projected point(s) of the same shape.
    """
    pts = np.array(pts)
    if pts.ndim == 1:
        pts = np.reshape(pts, (1, -1))
    # Guaranteed N-by-M
    basis = np.array(basis)

    w = np.dot(pts, basis) / (np.linalg.norm(basis) ** 2) # length N
    w = np.tile(w.reshape((-1, 1)), (1, len(basis))) # N-by-M

    proj = w * basis # N-by-M

    return proj


def calc_refl_vec(h, l):
    """Calculates the reflection vector given the half vector.

    Args:
        h (array_like): Half vector as a 3-array.
        l (array_like): "Incident" vector (pointing outwards from the surface
            point), as a 3-array.

    Returns:
        numpy.ndarray: Reflection vector as a 3-array.
    """
    h, l = np.array(h), np.array(l)
    h, l = normalize(h), normalize(l)
    v = 2 * (h @ l) * h - l
    return v


def solve_quadratic_eqn(a, b, c):
    """Solves :math:`ax^2+bx+c=0`.
    """
    x1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    x2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return x1, x2


def main(func_name):
    """Unit tests that can also serve as example usage."""
    if func_name == 'is_symmetric':
        mat = np.random.random((10, 9))
        print(is_symmetric(mat))
        mat = np.random.random((10, 10))
        print(is_symmetric(mat))
        mat = np.random.random((10, 10))
        mat = mat + mat.T
        print(is_symmetric(mat))
    else:
        raise NotImplementedError("Unit tests for %s" % func_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)
