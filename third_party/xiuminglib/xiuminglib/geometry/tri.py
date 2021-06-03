import numpy as np


def barycentric(pts, tvs):
    r"""Computes barycentric coordinates of 3D point(s) w.r.t. a triangle.

    Args:
        pts (array_like): 3-array for one point; N-by-3 array for multiple
            points.
        tvs (array_like): 3-by-3 array with rows being the triangle's
            vertices.

    Returns:
        numpy.ndarray: Barycentric coordinates of the same shape as ``pts``.
        If any array element :math:`\notin [0, 1]`, the input point doesn't
        fall on the triangle.
    """
    pts = np.array(pts)
    tvs = np.array(tvs)
    input_shape = pts.shape
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))

    vec0 = tvs[1] - tvs[0]
    vec1 = tvs[2] - tvs[0]
    vec2 = pts - tvs[0]
    d00 = vec0.dot(vec0)
    d01 = vec0.dot(vec1)
    d11 = vec1.dot(vec1)
    d20 = vec2.dot(vec0)
    d21 = vec2.dot(vec1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    uvw = np.hstack((u.reshape((-1, 1)),
                     v.reshape((-1, 1)),
                     w.reshape((-1, 1))))
    return uvw.reshape(input_shape)


def moeller_trumbore(ray_orig, ray_dir, tri_v0, tri_v1, tri_v2):
    r"""Decides if a ray intersects with a triangle using the Moeller-Trumbore
    algorithm.

    :math:`O + D = (1-u-v)V_0 + uV_1 + vV_2`.

    Args:
        ray_orig (array_like): 3D coordinates of the ray origin :math:`O`.
        ray_dir (array_like): Ray direction :math:`D` (not necessarily
            normalized).
        tri_v0 (array_like): Triangle vertex :math:`V_0`.
        tri_v1 (array_like): Triangle vertex :math:`V_1`.
        tri_v2 (array_like): Triangle vertex :math:`V_2`.

    Returns:
        tuple:
            - **u** (*float*) -- The :math:`u` component of the Barycentric
              coordinates of the intersection. Intersection is in-triangle
              (including on an edge or at a vertex), if :math:`u\geq 0`,
              :math:`v\geq 0`, and :math:`u+v\leq 1`.
            - **v** (*float*) -- The :math:`v` component.
            - **t** (*float*) -- Distance coefficient from :math:`O` to the
              intersection along :math:`D`. Intersection is between :math:`O`
              and :math:`O+D`, if :math:`0 < t < 1`.
    """
    # Validate inputs
    ray_orig = np.array(ray_orig)
    ray_dir = np.array(ray_dir)
    tri_v0 = np.array(tri_v0)
    tri_v1 = np.array(tri_v1)
    tri_v2 = np.array(tri_v2)
    assert (ray_orig.shape == (3,)), "'ray_orig' must be of length 3"
    assert (ray_dir.shape == (3,)), "'ray_dir' must be of length 3"
    assert (tri_v0.shape == (3,)), "'tri_v0' must be of length 3"
    assert (tri_v1.shape == (3,)), "'tri_v1' must be of length 3"
    assert (tri_v2.shape == (3,)), "'tri_v2' must be of length 3"

    M = np.array([-ray_dir, tri_v1 - tri_v0, tri_v2 - tri_v0]).T # noqa: N806
    y = (ray_orig - tri_v0).T
    t, u, v = np.linalg.solve(M, y)

    return u, v, t
