from copy import deepcopy
import numpy as np

from .imprt import preset_import

from .log import get_logger
logger = get_logger()


def normalize_uint(arr):
    r"""Normalizes the input ``uint`` array such that its ``dtype`` maximum
    becomes :math:`1`.

    Args:
        arr (numpy.ndarray): Input array of type ``uint``.

    Returns:
        numpy.ndarray: Normalized array of type ``float``.
    """
    if arr.dtype not in (np.uint8, np.uint16):
        raise TypeError(arr.dtype)
    maxv = np.iinfo(arr.dtype).max
    arr_ = arr.astype(float)
    arr_ = arr_ / maxv
    return arr_


def denormalize_float(arr, uint_type='uint8'):
    r"""De-normalizes the input ``float`` array such that :math:`1` becomes
    the target ``uint`` maximum.

    Args:
        arr (numpy.ndarray): Input array of type ``float``.
        uint_type (str, optional): Target ``uint`` type.

    Returns:
        numpy.ndarray: De-normalized array of the target type.
    """
    _assert_float_0to1(arr)
    if uint_type not in ('uint8', 'uint16'):
        raise TypeError(uint_type)
    maxv = np.iinfo(uint_type).max
    arr_ = arr * maxv
    arr_ = arr_.astype(uint_type)
    return arr_


def alpha_blend(arr1, alpha, arr2=None):
    r"""Alpha-blends two arrays, or masks one array.

    Args:
        arr1 (numpy.ndarray): Input array.
        alpha (numpy.ndarray): Alpha map whose values are :math:`\in [0,1]`.
        arr2 (numpy.ndarray): Input array. If ``None``, ``arr1`` will be
            blended with an all-zero array, equivalent to masking ``arr1``.

    Returns:
        numpy.ndarray: Blended array of type ``float``.
    """
    arr1 = arr1.astype(float)
    if arr2 is None:
        arr2 = np.zeros(arr1.shape, dtype=arr1.dtype)

    if alpha.shape != arr1.shape:
        if alpha.ndim == 2 and arr1.ndim == 3:
            alpha = np.dstack([alpha] * arr1.shape[2])
        elif alpha.ndim == 3 and alpha.shape[2] == 1 and arr1.ndim == 3:
            alpha = np.tile(alpha, (1, 1, arr1.shape[2]))
        else:
            raise NotImplementedError(
                "{arr_s} and {alpha_s}".format(
                    alpha_s=alpha.shape, arr_s=arr1.shape))
    blend = np.multiply(arr1, alpha) + np.multiply(arr2, 1 - alpha)
    return blend


def resize(arr, new_h=None, new_w=None, method='cv2'):
    """Resizes an image, with the option of maintaining the aspect ratio.

    Args:
        arr (numpy.ndarray): Image to binarize. If multiple-channel, each
            channel is resized independently.
        new_h (int, optional): Target height. If ``None``, will be calculated
            according to the target width, assuming the same aspect ratio.
        new_w (int, optional): Target width. If ``None``, will be calculated
            according to the target height, assuming the same aspect ratio.
        method (str, optional): Accepted values: ``'cv2'`` and ``'tf'``.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = arr.shape[:2]
    if new_h is not None and new_w is not None:
        if int(h / w * new_w) != new_h:
            logger.warning((
                "Aspect ratio changed in resizing: original size is %s; "
                "new size is %s"), (h, w), (new_h, new_w))
    elif new_h is None and new_w is not None:
        new_h = int(h / w * new_w)
    elif new_h is not None and new_w is None:
        new_w = int(w / h * new_h)
    else:
        raise ValueError("At least one of new height or width must be given")

    if method in ('tf', 'tensorflow'):
        tf = preset_import('tensorflow', assert_success=True)
        tf.compat.v1.enable_eager_execution()
        tensor = tf.convert_to_tensor(arr)
        tensor_resized = tf.image.resize(
            tensor, (new_h, new_w), method='bilinear', antialias=True)
        resized = tensor_resized.numpy()

    elif method in ('cv', 'cv2', 'opencv'):
        cv2 = preset_import('cv2', assert_success=True)
        interp = cv2.INTER_LINEAR if new_h > h else cv2.INTER_AREA
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    else:
        raise NotImplementedError(method)

    return resized


def binarize(im, threshold=None):
    """Binarizes images.

    Args:
        im (numpy.ndarray): Image to binarize. Of any integer type (``uint8``,
            ``uint16``, etc.).  If H-by-W-by-3, will be converted to grayscale
            and treated as H-by-W.
        threshold (float, optional): Threshold for binarization. ``None``
            means midpoint of the ``dtype``.

    Returns:
        numpy.ndarray: Binarized image. Of only 0's and 1's.
    """
    im_copy = deepcopy(im)

    # RGB to grayscale
    if im_copy.ndim == 3 and im_copy.shape[2] == 3: # h-by-w-by-3
        cv2 = preset_import('cv2', assert_success=True)
        im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)

    if im_copy.ndim == 2: # h-by-w
        # Compute threshold from data type
        if threshold is None:
            maxval = np.iinfo(im_copy.dtype).max
            threshold = maxval / 2.
        im_bin = im_copy
        logicalmap = im_copy > threshold
        im_bin[logicalmap] = 1
        im_bin[np.logical_not(logicalmap)] = 0
    else:
        raise ValueError("'im' is neither h-by-w nor h-by-w-by-3")

    return im_bin


def remove_islands(im, min_n_pixels, connectivity=4):
    """Removes small islands of pixels from a binary image.

    Args:
        im (numpy.ndarray): Input binary image. Of only 0's and 1's.
        min_n_pixels (int): Minimum island size to keep.
        connectivity (int, optional): Definition of "connected": either 4 or 8.

    Returns:
        numpy.ndarray: Output image with small islands removed.
    """
    cv2 = preset_import('cv2', assert_success=True)

    # Validate inputs
    assert (len(im.shape) == 2), \
        "'im' needs to have exactly two dimensions"
    assert np.array_equal(np.unique(im), np.array([0, 1])), \
        "'im' needs to contain only 0's and 1's"
    assert connectivity in (4, 8), \
        "'connectivity' must be either 4 or 8"

    # Find islands, big or small
    nlabels, labelmap, leftx_topy_bbw_bbh_npix, _ = \
        cv2.connectedComponentsWithStats(im, connectivity)

    # Figure out background is 0 or 1
    bgval = im[labelmap == 0][0]

    # Set small islands to background value
    im_clean = im
    for i in range(1, nlabels): # skip the 0th island -- background
        island_size = leftx_topy_bbw_bbh_npix[i, -1]
        if island_size < min_n_pixels:
            im_clean[labelmap == i] = bgval

    return im_clean


def grid_query_img(im, query_x, query_y, method='bilinear'):
    r"""Grid queries an image via interpolation.

    If you want to grid query unstructured data, consider
    :func:`grid_query_unstruct`.

    This function uses either bilinear interpolation that allows you to break
    big matrices into patches and work locally, or bivariate spline
    interpolation that fits a global spline (so memory-intensive) and shows
    global effects.

    Args:
        im (numpy.ndarray): H-by-W or H-by-W-by-C rectangular grid of data.
            Each of C channels is interpolated independently.
        query_x (array_like): :math:`x` coordinates of the queried rectangle,
            e.g., ``np.arange(10)`` for a 10-by-10 grid (hence, this should
            *not* be generated by :func:`numpy.meshgrid` or similar
            functions).
        query_y (array_like): :math:`y` coordinates, following this
            convention:

            .. code-block:: none

                +---------> query_x
                |
                |
                |
                v query_y

        method (str, optional): Interpolation method: ``'spline'`` or
            ``'bilinear'``.

    Returns:
        numpy.ndarray: Interpolated values at query locations, of shape
        ``(len(query_y), len(query_x))`` for single-channel input or
        ``(len(query_y), len(query_x), im.shape[2])`` for multi-channel
        input.
    """
    from scipy.interpolate import RectBivariateSpline, interp2d

    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        if c == 1: # single dimension
            im = im[:, :, 0]
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
    else:
        raise ValueError("'im' must have either two or three dimensions")

    x = np.arange(w)
    y = np.arange(h)

    if query_x.min() < 0 or query_x.max() > w - 1 or \
            query_y.min() < 0 or query_y.max() > h - 1:
        logger.warning("Sure you want to query points outside 'im'?")

    def query(x, y, z, qx, qy, method):
        if method == 'spline':
            # TODO: test whether we need to swap x and y
            spline_obj = RectBivariateSpline(y, x, z)
            qz = spline_obj(qy, qx, grid=True)
        elif method == 'bilinear':
            f = interp2d(x, y, z, kind='linear')
            qz = f(qx, qy)
        else:
            raise NotImplementedError("Other interplation methods")
        return qz

    if c == 1:
        # Single channel
        z = im
        logger.info("Interpolation (method: %s) started", method)
        interp_val = query(x, y, z, query_x, query_y, method)
        logger.info("... done")

    else:
        # Multiple channels
        interp_val = np.zeros((len(query_x), len(query_y), c))
        for i in range(c):
            z = im[:, :, i]
            logger.info(
                "Interpolation (method: %s) started for channel %d/%d",
                method, i + 1, c)
            interp_val[:, :, i] = query(x, y, z, query_x, query_y, method)
            logger.info("... done")

    return interp_val


def grid_query_unstruct(uvs, values, grid_res, method=None):
    r"""Grid queries unstructured data given by coordinates and their values.

    If you are looking to grid query structured data, such as an image, check
    out :func:`grid_query_img`.

    This function interpolates values on a rectangular grid given some sparse,
    unstrucured samples. One use case is where you have some UV locations and
    their associated colors, and you want to "paint the colors" on a UV canvas.

    Args:
        uvs (numpy.ndarray): N-by-2 array of UV coordinates where we have
            values (e.g., colors). See
            :func:`xiuminglib.blender.object.smart_uv_unwrap` for the UV
            coordinate convention.
        values (numpy.ndarray): N-by-M array of M-D values at the N UV
            locations, or N-array of scalar values at the N UV locations.
            Channels are interpolated independently.
        grid_res (array_like): Resolution (height first; then width) of
            the query grid.
        method (dict, optional): Dictionary of method-specific parameters.
            Implemented methods and their default parameters:

            .. code-block:: python

                # Default
                method = {
                    'func': 'griddata',
                    # Which SciPy function to call.

                    'func_underlying': 'linear',
                    # Fed to `griddata` as the `method` parameter.

                    'fill_value': (0,), # black
                    # Will be used to fill in pixels outside the convex hulls
                    # formed by the UV locations, and if `max_l1_interp` is
                    # provided, also the pixels whose interpolation is too much
                    # of a stretch to be trusted. In the context of "canvas
                    # painting," this will be the canvas' base color.

                    'max_l1_interp': np.inf, # trust/accept all interpolations
                    # Maximum L1 distance, which we can trust in interpolation,
                    # to pixels that have values. Interpolation across a longer
                    # range will not be trusted, and hence will be filled with
                    # `fill_value`.
                }

            .. code-block:: python

                method = {
                    'func': 'rbf',
                    # Which SciPy function to call.

                    'func_underlying': 'linear',
                    # Fed to `Rbf` as the `method` parameter.

                    'smooth': 0, # no smoothing
                    # Fed to `Rbf` as the `smooth` parameter.
                }

    Returns:
        numpy.ndarray: Interpolated values at query locations, of shape
        ``grid_res`` for single-channel input or ``(grid_res[0], grid_res[1],
        values.shape[2])`` for multi-channel input.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    assert values.ndim == 2 and values.shape[0] == uvs.shape[0]

    if method is None:
        method = {'func': 'griddata'}

    h, w = grid_res
    # Generate query coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # +---> x
    # |
    # v y
    grid_u, grid_v = grid_x, 1 - grid_y
    # ^ v
    # |
    # +---> u

    if method['func'] == 'griddata':
        from scipy.interpolate import griddata
        cv2 = preset_import('cv2', assert_success=True)

        func_underlying = method.get('func_underlying', 'linear')
        fill_value = method.get('fill_value', (0,))
        max_l1_interp = method.get('max_l1_interp', np.inf)

        fill_value = np.array(fill_value)
        if len(fill_value) == 1:
            fill_value = np.tile(fill_value, values.shape[1])
        assert len(fill_value) == values.shape[1]

        if max_l1_interp is None:
            max_l1_interp = np.inf # trust everything

        # Figure out which pixels can be trusted
        has_value = np.zeros((h, w), dtype=np.uint8)
        ri = ((1 - uvs[:, 1]) * (h - 1)).astype(int).ravel()
        ci = (uvs[:, 0] * (w - 1)).astype(int).ravel()
        in_canvas = np.logical_and.reduce(
            (ri >= 0, ri < h, ci >= 0, ci < w)) # to ignore out-of-canvas points
        has_value[ri[in_canvas], ci[in_canvas]] = 1
        dist2val = cv2.distanceTransform(1 - has_value, cv2.DIST_L1, 3)
        trusted = dist2val <= max_l1_interp

        # Process each color channel separately
        interps = []
        for ch_i in range(values.shape[1]):
            v_fill = fill_value[ch_i]
            v = values[:, ch_i]
            interp = griddata(uvs, v, (grid_u, grid_v),
                              method=func_underlying,
                              fill_value=v_fill)
            interp[~trusted] = v_fill
            interps.append(interp)
        interps = np.dstack(interps)

    elif method['func'] == 'rbf':
        from scipy.interpolate import Rbf

        func_underlying = method.get('func_underlying', 'linear')
        smooth = method.get('smooth', 0)

        # Process each color channel separately
        interps = []
        for ch_i in range(values.shape[1]):
            v = values[:, ch_i]
            rbfi = Rbf(uvs[:, 0], uvs[:, 1], v,
                       function=func_underlying,
                       smooth=smooth)
            interp = rbfi(grid_u, grid_v)
            interps.append(interp)
        interps = np.dstack(interps)

    else:
        raise NotImplementedError(method['func'])

    if interps.shape[2] == 1:
        return interps[:, :, 0].squeeze()
    return interps


def find_local_extrema(im, want_maxima, kernel_size=3):
    """Finds local maxima or minima in an image.

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale)
            or H-by-W-by-C for multi-channel (e.g., RGB) images. Extrema
            are found independently for each of the C channels.
        want_maxima (bool): Whether maxima or minima are wanted.
        kernel_size (int, optional): Side length of the square window under
            consideration. Must be larger than 1.

    Returns:
        numpy.ndarray: Binary map indicating if each pixel is a local extremum.
    """
    from scipy.ndimage.filters import minimum_filter, maximum_filter

    logger.error("find_local_extrema() not tested yet!")

    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        expanded = False
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
        im = np.expand_dims(im, axis=2) # adds singleton dimension
        expanded = True
    else:
        raise ValueError("'im' must have either two or three dimensions")

    kernel = np.ones((kernel_size, kernel_size)).astype(bool)

    is_extremum = np.zeros((h, w, c), dtype=bool)

    for i in range(c):
        z = im[:, :, i]

        if want_maxima:
            equals_extremum = maximum_filter(z, footprint=kernel) == z
        else:
            equals_extremum = minimum_filter(z, footprint=kernel) == z

        is_extremum[:, :, i] = equals_extremum

    if expanded:
        is_extremum = is_extremum[:, :, 0]

    return is_extremum


def compute_gradients(im):
    """Computes magnitudes and orientations of image gradients.

    With Scharr operators:

    .. code-block:: none

        [ 3 0 -3 ]           [ 3  10  3]
        [10 0 -10]    and    [ 0   0  0]
        [ 3 0 -3 ]           [-3 -10 -3]

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale) or
            H-by-W-by-C if multi-channel (e.g., RGB) images. Gradients are
            computed independently for each of the C channels.

    Returns:
        tuple:
            - **grad_mag** (*numpy.ndarray*) -- Magnitude image of the
              gradients.
            - **grad_orient** (*numpy.ndarray*) -- Orientation image of the
              gradients (in radians).

              .. code-block:: none

                       y ^ pi/2
                         |
                pi       |
                 --------+--------> 0
                -pi      |       x
                         | -pi/2
    """
    cv2 = preset_import('cv2', assert_success=True)

    # Figure out image size and number of channels
    if im.ndim == 3:
        h, w, c = im.shape
        expanded = False
    elif im.ndim == 2:
        h, w = im.shape
        c = 1
        im = np.expand_dims(im, axis=2) # adds singleton dimension
        expanded = True
    else:
        raise ValueError("'im' must have either two or three dimensions")

    grad_mag = np.zeros((h, w, c))
    grad_orient = np.zeros((h, w, c))

    for i in range(c):
        z = im[:, :, i]
        ddepth = -1 # same depth as the source

        # Along horizontal direction
        xorder, yorder = 1, 0
        grad_h = cv2.Sobel(z, ddepth, xorder, yorder, ksize=-1) # 3x3 Scharr
        grad_h = grad_h.astype(float)

        # Along vertical direction
        xorder, yorder = 0, 1
        grad_v = cv2.Sobel(z, ddepth, xorder, yorder, ksize=-1) # 3x3 Scharr
        grad_v = grad_v.astype(float)

        # Magnitude
        grad_mag[:, :, i] = np.sqrt(np.square(grad_h) + np.square(grad_v))

        # Orientation
        grad_orient[:, :, i] = np.arctan2(grad_v, grad_h)

    if expanded:
        grad_mag = grad_mag[:, :, 0]
        grad_orient = grad_orient[:, :, 0]

    return grad_mag, grad_orient


def gamma_correct(im, gamma=2.2):
    r"""Applies gamma correction to an ``uint`` image.

    Args:
        im (numpy.ndarray): H-by-W if single-channel (e.g., grayscale) or
            H-by-W-by-C multi-channel (e.g., RGB) ``uint`` images.
        gamma (float, optional): Gamma value :math:`< 1` shifts image towards
            the darker end of the spectrum, while value :math:`> 1` towards
            the brighter.

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    cv2 = preset_import('cv2', assert_success=True)
    assert im.dtype in ('uint8', 'uint16')

    # Don't correct alpha channel, if exists
    alpha = None
    if im.ndim == 3 and im.shape[2] == 4:
        alpha = im[:, :, 3]
        im = im[:, :, :3]

    # Correct with lookup table
    type_max = np.iinfo(im.dtype).max
    table = np.array([
        ((x / type_max) ** (1 / gamma)) * type_max
        for x in np.arange(0, type_max + 1)
    ]).astype(im.dtype)
    im_corrected = cv2.LUT(im, table)

    # Concat alpha channel back
    if alpha is not None:
        im_corrected = np.dstack((im_corrected, alpha))

    return im_corrected


def rgb2lum(im):
    """Converts RGB to relative luminance (if input is linear RGB) or luma
    (if input is gamma-corrected RGB).

    Args:
        im (numpy.ndarray): RGB array of shape ``(..., 3)``.

    Returns:
        numpy.ndarray: Relative luminance or luma array.
    """
    assert im.shape[-1] == 3, "Input's last dimension must hold RGB"

    lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]

    return lum


def _assert_float_0to1(arr):
    if arr.dtype.kind != 'f':
        raise TypeError("Input must be float (is %s)" % arr.dtype)
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("Input image has pixels outside [0, 1]")


def _assert_3ch(arr):
    if arr.ndim != 3:
        raise ValueError("Input image is not even 3D (H-by-W-by-3)")
    n_ch = arr.shape[2]
    if n_ch != 3:
        raise ValueError("Input image must have 3 channels, but has %d" % n_ch)


srgb_linear_thres = 0.0031308
srgb_linear_coeff = 12.92
srgb_exponential_coeff = 1.055
srgb_exponent = 2.4


def linear2srgb(im, clip=False):
    r"""Converts an image from linear RGB values to sRGB.

    Args:
        im (numpy.ndarray): Of type ``float``, and all pixels must be
            :math:`\in [0, 1]`.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Returns:
        numpy.ndarray: Converted image in sRGB.
    """
    if clip:
        im = np.clip(im, 0, 1)
    _assert_float_0to1(im)
    im_ = deepcopy(im)
    # Guaranteed to be [0, 1] floats

    linear_ind = im_ <= srgb_linear_thres
    nonlinear_ind = im_ > srgb_linear_thres
    im_[linear_ind] = im_[linear_ind] * srgb_linear_coeff
    im_[nonlinear_ind] = srgb_exponential_coeff * (
        np.power(im_[nonlinear_ind], 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    return im_


def srgb2linear(im, clip=False):
    r"""Converts an image from sRGB values to linear RGB.

    Args:
        im (numpy.ndarray): Of type ``float``, and all pixels must be
            :math:`\in [0, 1]`.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Returns:
        numpy.ndarray: Converted image in linear RGB.
    """
    if clip:
        im = np.clip(im, 0, 1)
    _assert_float_0to1(im)
    im_ = deepcopy(im)
    # Guaranteed to be [0, 1] floats

    gamma = (
        (im_ + srgb_exponential_coeff - 1) / srgb_exponential_coeff
    ) ** srgb_exponent
    scale = im_ / srgb_linear_coeff
    im_ = np.where(im_ > srgb_linear_thres * srgb_linear_coeff, gamma, scale)

    return im_


def tonemap(hdr, method='gamma', gamma=2.2):
    r"""Tonemaps an HDR image.

    Args:
        hdr (numpy.ndarray): HDR image.
        method (str, optional): Values accepted: ``'gamma'`` and ``'reinhard'``.
        gamma (float, optional): Gamma value used if method is ``'gamma'``.

    Returns:
        numpy.ndarray: Tonemapped image :math:`\in [0, 1]`.
    """
    if method == 'reinhard':
        cv2 = preset_import('cv2', assert_success=True)
        tonemapper = cv2.createTonemapReinhard(1, 1, 0, 0)
        img = tonemapper.process(hdr)
    elif method == 'gamma':
        img = (hdr / hdr.max()) ** (1 / gamma)
    else:
        raise ValueError(method)

    # Clip, if necessary, to guard against numerical errors
    minv, maxv = img.min(), img.max()
    if minv < 0:
        logger.warning("Clipping negative values (min.: %f)", minv)
        img = np.clip(img, 0, np.inf)
    if maxv > 1:
        logger.warning("Clipping >1 values (max.: %f)", maxv)
        img = np.clip(img, -np.inf, 1)

    return img
