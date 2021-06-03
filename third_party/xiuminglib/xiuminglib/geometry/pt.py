import numpy as np


def ptcld2tdf(pts, res=128, center=False):
    """Converts point cloud to truncated distance function (TDF).

    Maximum distance is capped at 1 / ``res``.

    Args:
        pts (array_like): Cartesian coordinates in object space. Of shape
            N-by-3.
        res (int, optional): Resolution of the TDF.
        center (bool, optional): Whether to center these points around the
            object space origin.

    Returns:
        numpy.ndarray: Output TDF.
    """
    pts = np.array(pts)

    n_pts = pts.shape[0]

    if center:
        pts_center = np.mean(pts, axis=0)
        pts -= np.tile(pts_center, (n_pts, 1))

    tdf = np.ones((res, res, res)) / res
    cnt = np.zeros((res, res, res))

    # -0.5 to 0.5 in every dimension
    extent = 2 * np.abs(pts).max()
    pts_scaled = pts / extent

    # Compute distance from center of each involved voxel to its surface
    # points
    for i in range(n_pts):
        pt = pts_scaled[i, :]
        ind = np.floor((pt + 0.5) * (res - 1)).astype(int)
        v_ctr = (ind + 0.5) / (res - 1) - 0.5
        dist = np.linalg.norm(pt - v_ctr)
        n = cnt[ind[0], ind[1], ind[2]]
        tdf[ind[0], ind[1], ind[2]] = \
            (tdf[ind[0], ind[1], ind[2]] * n + dist) / (n + 1)
        cnt[ind[0], ind[1], ind[2]] += 1

    return tdf
