import numpy as np

from .rot import _warn_degree

from ..log import get_logger
logger = get_logger()


def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    n_ = np.sqrt(n)
    if n_ != int(n_):
        raise ValueError("%d is not perfect square" % n)
    n_ = int(n_)

    pts_r_theta_phi = []
    for u in np.linspace(0, 1, n_):
        for v in np.linspace(0, 1, n_):
            theta = np.arccos(2 * u - 1) # [0, pi]
            phi = 2 * np.pi * v # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))
    pts_r_theta_phi = np.vstack(pts_r_theta_phi)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions(
            pts_r_theta_phi, 'theta-phi_to_lat-lng')
    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = np.array(pts_cart)

    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph


def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def main(func_name):
    """Unit tests that can also serve as example usage."""
    if func_name in ('sph2cart', 'cart2sph'):
        # cart2sph() and sph2cart()
        pts_car = np.array([
            [-1, 2, 3],
            [4, -5, 6],
            [3, 5, -8],
            [-2, -5, 2],
            [4, -2, -23]])
        print(pts_car)
        pts_sph = cart2sph(pts_car)
        print(pts_sph)
        pts_car_recover = sph2cart(pts_sph)
        print(pts_car_recover)
    else:
        raise NotImplementedError("Unit tests for %s" % func_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)
