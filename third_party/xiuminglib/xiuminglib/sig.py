import numpy as np

from .imprt import preset_import


def get_extrema(arr, top=True, n=1, n_std=None):
    """Gets top (or bottom) N value(s) from an M-D array, with the option to
    ignore outliers.

    Args:
        arr (array_like): Array, which will be flattened if high-D.
        top (bool, optional): Whether to find the top or bottom N.
        n (int, optional): Number of values to return.
        n_std (float, optional): Definition of outliers to exclude, assuming
            Gaussian. ``None`` means assuming no outlier.

    Returns:
        tuple:
            - **ind** (*tuple*) -- Indices that give the extrema, M-tuple of
              arrays of N integers.
            - **val** (*numpy.ndarray*) -- Extremum values, i.e.,
              ``arr[ind]``.
    """
    arr = np.array(arr, dtype=float)

    if top:
        arr_to_sort = -arr.flatten()
    else:
        arr_to_sort = arr.flatten()

    if n_std is not None:
        meanv = np.mean(arr_to_sort)
        stdv = np.std(arr_to_sort)
        arr_to_sort[np.logical_or(
            arr_to_sort < meanv - n_std * stdv,
            arr_to_sort > meanv + n_std * stdv,
        )] = np.nan # considered greater than numbers

    ind = [
        x for x in np.argsort(arr_to_sort)
        if not np.isnan(arr_to_sort[x])
    ][:n] # 1D indices
    ind = np.unravel_index(ind, arr.shape) # Back to high-D
    val = arr[ind]

    return ind, val


def smooth_1d(arr, win_size, kernel_type='half'):
    """Smooths 1D signal.

    Args:
        arr (array_like): 1D signal to smooth.
        win_size (int): Size of the smoothing window. Use odd number.
        kernel_type (str, optional): Kernel type: ``'half'`` (e.g., normalized
            :math:`[2^{-2}, 2^{-1}, 2^0, 2^{-1}, 2^{-2}]`) or ``'equal'``
            (e.g., normalized :math:`[1, 1, 1, 1, 1]`).

    Returns:
        numpy.ndarray: Smoothed 1D signal.
    """
    assert np.mod(win_size, 2) == 1, "Even window size provided"
    arr = np.array(arr).ravel()

    # Generate kernel
    if kernel_type == 'half':
        kernel = np.array(
            [2 ** x if x < 0 else 2 ** -x
             for x in range(-int(win_size / 2), int(win_size / 2) + 1)])
    elif kernel_type == 'equal':
        kernel = np.ones(win_size)
    else:
        raise ValueError("Unidentified kernel type")
    kernel /= sum(kernel)
    n = (win_size - 1) // 2

    arr_pad = np.hstack((arr[0] * np.ones(n), arr, arr[-1] * np.ones(n)))
    arr_smooth = np.convolve(arr_pad, kernel, 'valid')

    # Restore original values of the head and tail
    arr_smooth[0] = arr[0]
    arr_smooth[-1] = arr[-1]

    return arr_smooth


def pca(data_mat, n_pcs=None, eig_method='scipy.sparse.linalg.eigsh'):
    """Performs principal component (PC) analysis on data.

    Via eigendecomposition of covariance matrix. See :func:`main` for example
    usages, including reconstructing data with top K PCs.

    Args:
        data_mat (array_like): Data matrix of N data points in the M-D space,
            of shape M-by-N, where each column is a point.
        n_pcs (int, optional): Number of top PCs requested. ``None`` means
            :math:`M-1`.
        eig_method (str, optional): Method for eigendecomposition of the
            symmetric covariance matrix: ``'numpy.linalg.eigh'`` or
            ``'scipy.sparse.linalg.eigsh'``.

    Returns:
        tuple:
            - **pcvars** (*numpy.ndarray*) -- PC variances (eigenvalues of
              covariance matrix) in descending order.
            - **pcs** (*numpy.ndarray*) -- Corresponding PCs (normalized
              eigenvectors), of shape M-by-``n_pcs``. Each column is a PC.
            - **projs** (*numpy.ndarray*) -- Data points centered and then
              projected to the ``n_pcs``-D PC space. Of shape ``n_pcs``-by-N.
              Each column is a point.
            - **data_mean** (*numpy.ndarray*) -- Mean that can be used to
              recover raw data. Of length M.
    """
    from scipy.sparse import issparse
    from scipy.sparse.linalg import eigsh

    if issparse(data_mat):
        data_mat = data_mat.toarray()
    else:
        data_mat = np.array(data_mat)
    # data_mat is NOT centered

    if n_pcs is None:
        n_pcs = data_mat.shape[0] - 1

    # ------ Compute covariance matrix of data

    covmat = np.cov(data_mat) # auto handles uncentered data
    # covmat is real and symmetric in theory, but may not be so
    # due to numerical issues, so eigendecomposition method should be told
    # explicitly to exploit symmetry constraints

    # ------ Compute eigenvalues and eigenvectors

    if eig_method == 'scipy.sparse.linalg.eigsh':
        # Largest (in magnitude) n_pcs eigenvalues
        eig_vals, eig_vecs = eigsh(covmat, k=n_pcs, which='LM')
        # eig_vals in ascending order
        # eig_vecs columns are normalized eigenvectors

        pcvars = eig_vals[::-1] # descending
        pcs = eig_vecs[:, ::-1]

    elif eig_method == 'numpy.linalg.eigh':
        # eigh() prevents complex eigenvalues, compared with eig()
        eig_vals, eig_vecs = np.linalg.eigh(covmat)
        # eig_vals in ascending order
        # eig_vecs columns are normalized eigenvectors

        # FIXME: sometimes the eigenvalues are not sorted? Subnormals appear
        # All zero eigenvectors
        sort_ind = eig_vals.argsort() # ascending
        eig_vals = eig_vals[sort_ind]
        eig_vecs = eig_vecs[:, sort_ind]

        pcvars = eig_vals[:-(n_pcs + 1):-1] # descending
        pcs = eig_vecs[:, :-(n_pcs + 1):-1]

    else:
        raise NotImplementedError(eig_method)

    # ------ Center and then project data points to PC space

    data_mean = np.mean(data_mat, axis=1)
    data_mat_centered = data_mat - np.tile(data_mean.reshape(-1, 1),
                                           (1, data_mat.shape[1]))
    projs = np.dot(pcs.T, data_mat_centered)

    return pcvars, pcs, projs, data_mean


def dct_1d_bases(n):
    """Generates 1D discrete cosine transform (DCT) bases.

    Bases are rows of :math:`Y`, which is orthogonal: :math:`Y^TY=YY^T=I`.
    The forward process (analysis) is :math:`X=Yx`, and the inverse
    (synthesis) is :math:`x=Y^{-1}X=Y^TX`. See :func:`main` for example usages
    and how this produces the same results as :func:`scipy.fftpack.dct` (with
    ``type=2`` and ``norm='ortho'``).

    Args:
        n (int): Signal length.

    Returns:
        numpy.ndarray: Matrix whose :math:`i`-th row, when dotted with signal
        (column) vector, gives the coefficient for the :math:`i`-th DCT
        component. Of shape ``(n, n)``.
    """
    col_ind, row_ind = np.meshgrid(range(n), range(n))
    omega = np.multiply(row_ind, (2 * col_ind + 1) / (2 * n) * np.pi)
    wmat = np.cos(omega)
    wmat[0, :] = wmat[0, :] / np.sqrt(2)
    wmat = np.sqrt(2 / n) * wmat # normalize so that orthogonal
    return wmat


def dct_2d_bases(h, w):
    r"""Generates bases for 2D discrete cosine transform (DCT).

    Bases are given in two matrices :math:`Y_h` and :math:`Y_w`. See
    :func:`dct_1d_bases` for their properties. Note that :math:`Y_w` has
    already been transposed (hence, :math:`Y_hxY_w` instead of
    :math:`Y_hxY_w^T` below).

    Input image :math:`x` should be transformed by both matrices (i.e., along
    both dimensions). Specifically, the analysis process is :math:`X=Y_hxY_w`,
    and the synthesis process is :math:`x=Y_h^TXY_w^T`. See :func:`main` for
    example usages.

    Args:
        h (int): Image height.
        w

    Returns:
        tuple:
            - **dct_mat_h** (*numpy.ndarray*) -- DCT matrix :math:`Y_h`
              transforming rows of the 2D signal. Of shape ``(h, h)``.
            - **dct_mat_w** (*numpy.ndarray*) -- :math:`Y_w` transforming\
              columns. Of shape ``(w, w)``.
    """
    dct_mat_h = dct_1d_bases(h)
    dct_mat_w = dct_1d_bases(w).T
    return dct_mat_h, dct_mat_w


def dct_2d_bases_vec(h, w):
    r"""Generates bases stored in a single matrix, along whose height 2D
    frequencies get raveled.

    Using the "vectorization + Kronecker product" trick:
    :math:`\operatorname{vec}(Y_hxY_w)=\left(Y_w^T\otimes Y_h\right)
    \operatorname{vec}(x)`. So unlike :func:`dct_2d_bases`, this function
    generates a single matrix :math:`Y=Y_w^T\otimes Y_h`, whose :math:`k`-th
    row is the flattened :math:`(i, j)`-th basis, where :math:`k=wi+j`.

    Input image :math:`x` can be transformed with a single matrix
    multiplication. Specifically, the analysis process is :math:`X=Y
    \operatorname{vec}(x)`, and the synthesis process is :math:`x=
    \operatorname{unvec}(Y^TX)`. See :func:`main` for examples.

    Warning:
        If you want to reconstruct the signal with only some (i.e., not all)
        bases, do not slice those rows out from :math:`Y` and use only their
        coefficients. Instead, you should use the full :math:`Y` matrix and
        set to zero the coefficients for the unused frequency components.
        See :func:`main` for examples.

    Args:
        h (int): Image height.
        w

    Returns:
        numpy.ndarray: Matrix with flattened bases as rows. The :math:`k`-th
        row, when :func:`numpy.reshape`'ed into ``(h, w)``, is the :math:`
        (i, j)`-th frequency component, where :math:`k=wi+j`. Of shape
        ``(h * w, h * w)``.
    """
    dct_mat_h, dct_mat_w = dct_2d_bases(h, w)
    dct_mat = np.kron(dct_mat_w.T, dct_mat_h)
    return dct_mat


def dft_1d_bases(n):
    """Generates 1D discrete Fourier transform (DFT) bases.

    Bases are rows of :math:`Y`, which is unitary (:math:`Y^HY=YY^H=I`,
    where :math:`Y^H` is the conjugate transpose) and symmetric. The forward
    process (analysis) is :math:`X=Yx`, and the inverse (synthesis) is
    :math:`x=Y^{-1}X=Y^HX`. See :func:`main` for example usages.

    Args:
        n (int): Signal length.

    Returns:
        numpy.ndarray: Matrix whose :math:`i`-th row, when dotted with signal
        (column) vector, gives the coefficient for the :math:`i`-th Fourier
        component. Of shape ``(n, n)``.
    """
    col_ind, row_ind = np.meshgrid(range(n), range(n))
    omega = np.exp(-2 * np.pi * 1j / n)
    wmat = np.power(omega, col_ind * row_ind) / np.sqrt(n)
    # Normalize so that unitary
    return wmat


def dft_2d_freq(h, w):
    """Gets 2D discrete Fourier transform (DFT) sample frequencies.

    Args:
        h (int): Image height.
        w

    Returns:
        tuple:
            - **freq_h** (*numpy.ndarray*) -- Sample frequencies, in cycles
              per pixel, along the height dimension. E.g., if ``freq_h[i, j]
              == 0.5``, then the ``(i, j)``-th component repeats every 2
              pixels along the height dimension.
            - **freq_w**
    """
    freq_h = np.fft.fftfreq(h)
    freq_w = np.fft.fftfreq(w)
    freq_h, freq_w = np.meshgrid(freq_h, freq_w, indexing='ij')
    return freq_h, freq_w


def dft_2d_bases(h, w):
    r"""Generates bases for 2D discrete Fourier transform (DFT).

    Bases are given in two matrices :math:`Y_h` and :math:`Y_w`. See
    :func:`dft_1d_bases` for their properties. Note that :math:`Y_w` has
    already been transposed.

    Input image :math:`x` should be transformed by both matrices (i.e., along
    both dimensions). Specifically, the analysis process is :math:`X=Y_hxY_w`,
    and the synthesis process is :math:`x=Y_h^HXY_w^H`. See :func:`main` for
    example usages and how this produces the same results as
    :func:`numpy.fft.fft2` (with ``norm='ortho'``).

    See Also:
        From :mod:`numpy.fft` -- ``A[1:n/2]`` contains the positive-frequency
        terms, and ``A[n/2+1:]`` contains the negative-frequency terms, in
        order of decreasingly negative frequency. For an even number of input
        points, ``A[n/2]`` represents both positive and negative Nyquist
        frequency, and is also purely real for real input. For an odd number
        of input points, ``A[(n-1)/2]`` contains the largest positive
        frequency, while ``A[(n+1)/2]`` contains the largest negative
        frequency.

    Args:
        h (int): Image height.
        w

    Returns:
        tuple:
            - **dft_mat_h** (*numpy.ndarray*) -- DFT matrix :math:`Y_h`
              transforming rows of the 2D signal. Of shape ``(h, h)``.
            - **dft_mat_w** (*numpy.ndarray*) -- :math:`Y_w` transforming
              columns. Of shape ``(w, w)``.
    """
    dft_mat_h = dft_1d_bases(h)
    dft_mat_w = dft_1d_bases(w).T # shouldn't matter, as it's symmetric
    return dft_mat_h, dft_mat_w


def dft_2d_bases_vec(h, w):
    r"""Generates bases stored in a single matrix, along whose height 2D
    frequencies get raveled.

    Using the "vectorization + Kronecker product" trick:
    :math:`\operatorname{vec}(Y_hxY_w)=\left(Y_w^T\otimes Y_h\right)
    \operatorname{vec}(x)`. So unlike :func:`dft_2d_bases`, this function
    generates a single matrix :math:`Y=Y_w^T\otimes Y_h`, whose :math:`k`-th
    row is the flattened :math:`(i, j)`-th basis, where :math:`k=wi+j`.

    Input image :math:`x` can be transformed with a single matrix
    multiplication. Specifically, the analysis process is
    :math:`X=Y\operatorname{vec}(x)`, and the synthesis process is
    :math:`x=\operatorname{unvec}(Y^HX)`. See :func:`main` for examples.

    Args:
        h (int): Image height.
        w

    Returns:
        numpy.ndarray: Complex matrix with flattened bases as rows. The
        :math:`k`-th row, when :func:`numpy.reshape`'ed into ``(h, w)``, is
        the :math:`(i, j)`-th frequency component, where :math:`k=wi+j`.
        Of shape ``(h * w, h * w)``.
    """
    dft_mat_h, dft_mat_w = dft_2d_bases(h, w)
    dft_mat = np.kron(dft_mat_w.T, dft_mat_h)
    return dft_mat


def sh_bases_real(l, n_lat, coord_convention='colatitude-azimuth', _check_orthonormality=False):
    r"""Generates real spherical harmonics (SHs).

    See :func:`main` for example usages, including how to do both analysis and
    synthesis the SHs.

    Not accurate when ``n_lat`` is too small. E.g., orthonormality no longer
    holds when discretization is too coarse (small ``n_lat``), as numerical
    integration fails to approximate the continuous integration.

    Args:
        l (int): Up to which band (starting form 0). The number of harmonics
            is :math:`(l+1)^2`. In other words, all harmonics within each band
            (:math:`-l\leq m\leq l`) are used.
        n_lat (int): Number of discretization levels of colatitude (for
            colatitude-azimuth convention; :math:`[0, \pi]`) or latitude (for
            latitude-longitude convention; :math:`[-\frac{\pi}{2},
            \frac{\pi}{2}]`). With the same step size, ``n_azimuth`` will be
            twice as big, since azimuth (in colatitude-azimuth convention;
            :math:`[0, 2\pi]`) or latitude (in latitude-longitude convention;
            :math:`[-\pi, \pi]`) spans :math:`2\pi`.
        coord_convention (str, optional): Coordinate system convention to use:
            ``'colatitude-azimuth'`` or ``'latitude-longitude'``.
            Colatitude-azimuth vs. latitude-longitude convention:

            .. code-block:: none

                3D
                                                   ^ z (colat = 0; lat = pi/2)
                                                   |
                          (azi = 3pi/2;            |
                           lng = -pi/2)   ---------+---------> y (azi = pi/2;
                                                 ,'|              lng = pi/2)
                                               ,'  |
                    (colat = pi/2, azi = 0;  x     | (colat = pi; lat = -pi/2)
                     lat = 0, lng = 0)

                2D
                    (0, 0)                               (pi/2, 0)
                       +----------->  (0, 2pi)               ^ lat
                       |            azi                      |
                       |                                     |
                       |                     (0, -pi) -------+-------> (0, pi)
                       v colat                               |        lng
                    (pi, 0)                                  |
                                                        (-pi/2, 0)

        _check_orthonormality (bool, optional, internal): Whether to check
            orthonormal or not.

    Returns:
        tuple:
            - **ymat** (*numpy.ndarray*) -- Matrix whose rows are spherical
              harmonics as generated by :func:`scipy.special.sph_harm`. When
              dotted with flattened image (column) vector weighted by
              ``areas_on_unit_sphere``, the :math:`i`-th row gives the
              coefficient for the :math:`i`-th harmonics, where
              :math:`i=l(l+1)+m`. The input signal (in the form of 2D image
              indexed by two angles) should be flattened with
              :meth:`numpy.ndarray.ravel`, in row-major order: the row index
              varies the slowest, and the column index the quickest. Of shape
              ``((l + 1) ** 2, 2 * n_lat ** 2)``.
            - **areas_on_unit_sphere** (*numpy.ndarray*) -- Area of the unit
              sphere covered by each sample point. This is proportional to
              sine of colatitude and has nothing to do with azimuth/longitude.
              Used as weights for discrete summation to approximate continuous
              integration. Necessary in SH analysis. Flattened also in
              row-major order. Of length ``n_lat * (2 * n_lat)``.
    """
    from scipy.special import sph_harm

    # Generate the l and m values for each matrix location
    l_mat = np.zeros(((l + 1) ** 2, n_lat * 2 * n_lat))
    m_mat = np.zeros(l_mat.shape)
    i = 0
    for curr_l in range(l + 1):
        for curr_m in range(-curr_l, curr_l + 1):
            l_mat[i, :] = curr_l * np.ones(l_mat.shape[1])
            m_mat[i, :] = curr_m * np.ones(l_mat.shape[1])
            i += 1

    # Generate the two angles for each matrix location
    if coord_convention == 'colatitude-azimuth':
        azis, colats = np.meshgrid(
            np.linspace(0, 2 * np.pi, num=2 * n_lat, endpoint=False),
            np.linspace(0, np.pi, num=n_lat, endpoint=True))
    elif coord_convention == 'latitude-longitude':
        lngs, lats = np.meshgrid(
            np.linspace(-np.pi, np.pi, num=2 * n_lat, endpoint=False),
            np.linspace(np.pi / 2, -np.pi / 2, num=n_lat, endpoint=True))
        colats = np.pi / 2 - lats
        azis = lngs
        azis[azis < 0] += 2 * np.pi
    else:
        raise NotImplementedError(coord_convention)

    # Evaluate (complex) SH at these locations
    colat_mat = np.tile(colats.ravel(), (l_mat.shape[0], 1))
    azi_mat = np.tile(azis.ravel(), (l_mat.shape[0], 1))
    ymat_complex = sph_harm(m_mat, l_mat, azi_mat, colat_mat)

    sin_colat = np.sin(colats.ravel())
    # Area on the unit sphere covered by each sample point, proportional to
    # sin(colat). Used as weights for discrete summation, approximating
    # continuous integration
    areas_on_unit_sphere = 4 * np.pi * sin_colat / np.sum(sin_colat)

    # Verify orthonormality of SHs
    if _check_orthonormality:
        print("Verifying Orthonormality of Complex SH Bases")
        print("(l1, m1) and (l2, m2):\treal\timag")
        for l1 in range(l + 1):
            for m1 in range(-l1, l1 + 1, 1):
                i1 = l1 * (l1 + 1) + m1
                y1 = ymat_complex[i1, :]
                for l2 in range(l + 1):
                    for m2 in range(-l2, l2 + 1, 1):
                        i2 = l2 * (l2 + 1) + m2
                        y2 = ymat_complex[i2, :]
                        integral = np.conj(y1).dot(
                            np.multiply(areas_on_unit_sphere, y2))
                        integral_real = np.real(integral)
                        integral_imag = np.imag(integral)
                        if np.isclose(integral_real, 0):
                            integral_real = 0
                        if np.isclose(integral_imag, 0):
                            integral_imag = 0
                        print("(%d, %d) and (%d, %d):\t%f\t%f" %
                              (l1, m1, l2, m2, integral_real, integral_imag))

    # Derive real SH's
    ymat_complex_real = np.real(ymat_complex)
    ymat_complex_imag = np.imag(ymat_complex)
    ymat = np.zeros(ymat_complex_real.shape)
    ind = m_mat > 0
    ymat[ind] = (-1) ** m_mat[ind] * np.sqrt(2) * ymat_complex_real[ind]
    ind = m_mat == 0
    ymat[ind] = ymat_complex_real[ind]
    ind = m_mat < 0
    ymat[ind] = (-1) ** m_mat[ind] * np.sqrt(2) * ymat_complex_imag[ind]

    if _check_orthonormality:
        print("Verifying Orthonormality of Real SH Bases")
        print("(l1, m1) and (l2, m2):\tvalue")
        for l1 in range(l + 1):
            for m1 in range(-l1, l1 + 1, 1):
                i1 = l1 * (l1 + 1) + m1
                y1 = ymat[i1, :]
                for l2 in range(l + 1):
                    for m2 in range(-l2, l2 + 1, 1):
                        i2 = l2 * (l2 + 1) + m2
                        y2 = ymat[i2, :]
                        integral = y1.dot(
                            np.multiply(areas_on_unit_sphere, y2))
                        if np.isclose(integral, 0):
                            integral = 0
                        print("(%d, %d) and (%d, %d):\t%f" %
                              (l1, m1, l2, m2, integral))

    return ymat, areas_on_unit_sphere


def main(test_id):
    """Unit tests that can also serve as example usage."""
    if test_id == 'pca':
        pts = np.random.rand(5, 8) # 8 points in 5D
        # Find all principal components
        n_pcs = pts.shape[0] - 1
        _, pcs, projs, data_mean = pca(pts, n_pcs=n_pcs)
        # Reconstruct data with only the top two PC's
        k = 2
        pts_recon = pcs[:, :k].dot(projs[:k, :]) + \
            np.tile(data_mean, (projs.shape[1], 1)).T
        print("Recon:")
        print(pts_recon)

    elif test_id == 'dct_1d_bases':
        from scipy.fftpack import dct
        from . import linalg
        signal = np.random.randint(0, 255, 8)
        n = len(signal)
        # Transform by my matrix
        dct_mat = dct_1d_bases(n)
        assert linalg.is_identity(dct_mat.T.dot(dct_mat), eps=1e-10)
        coeffs = dct_mat.dot(signal)
        recon = dct_mat.T.dot(coeffs)
        print("Max. difference between original and recon.: %e"
              % np.abs(signal - recon).max())
        # Transform by SciPy
        coeffs_sp = dct(signal, norm='ortho')
        print("Max. magnitude difference: %e"
              % np.abs(coeffs - coeffs_sp).max())

    elif test_id == 'dct_cameraman':
        from os.path import join
        from copy import deepcopy
        from scipy.fftpack import dct, idct
        from . import const, os as xm_os
        from .vis.matrix import matrix_as_heatmap, matrix_as_heatmap_complex
        cv2 = preset_import('cv2', assert_success=True)
        outdir = join(const.Dir.tmp, test_id)
        xm_os.makedirs(outdir, rm_if_exists=True)
        im = cv2.imread(const.Path.cameraman, cv2.IMREAD_GRAYSCALE) # TODO: switch to xm.io.img
        im = cv2.resize(im, (64, 64))
        cv2.imwrite(join(outdir, 'orig.png'), im) # TODO: switch to xm.io.img
        # Transform by my DCT (2-step)
        dct_mat_h, dct_mat_w = dct_2d_bases(*im.shape)
        coeffs_2step = dct_mat_h.dot(im).dot(dct_mat_w)
        recon_2step = dct_mat_h.T.dot(coeffs_2step).dot(dct_mat_w.T)
        cv2.imwrite(join(outdir, 'recon_2step.png'), recon_2step) # TODO: switch to xm.io.img
        print("(Ours 2-Step Full Recon. vs. Orig.) Max. difference: %e" %
              np.abs(im - recon_2step).max())
        # Transform by my DCT (1-step)
        dct_mat = dct_2d_bases_vec(*im.shape)
        coeffs_1step = dct_mat.dot(im.ravel())
        recon_1step = dct_mat.T.dot(coeffs_1step)
        cv2.imwrite(join(outdir, 'recon_1step.png'),
                    recon_1step.reshape(im.shape)) # TODO: switch to xm.io.img
        print("(Ours 1-Step Full Recon. vs. Orig.) Max. difference: %e" %
              np.abs(im - recon_1step.reshape(im.shape)).max())
        # for i in range(dct_mat.shape[0]):
        #     xm_vis.matrix_as_image(
        #         dct_mat[i, :].reshape(im.shape), outpath=join(outdir, 'basis%06d.png' % i))
        coeffs_1step_quarter = deepcopy(coeffs_1step).reshape(
            coeffs_2step.shape)
        coeffs_1step_quarter[(coeffs_1step_quarter.shape[0] // 2):,
                             (coeffs_1step_quarter.shape[1] // 2):] = 0
        coeffs_1step_quarter = coeffs_1step_quarter.ravel()
        recon_1step_quarter = dct_mat.T.dot(coeffs_1step_quarter)
        cv2.imwrite(join(outdir, 'recon_1step_quarter.png'),
                    recon_1step_quarter.reshape(im.shape)) # TODO: switch to xm.io.img
        # Transform by SciPy
        coeffs_sp = dct(
            dct(im.T, type=2, norm='ortho').T,
            type=2, norm='ortho')
        matrix_as_heatmap(
            coeffs_2step - coeffs_sp, outpath=join(outdir, '2step-sp.png'))
        matrix_as_heatmap(
            coeffs_1step.reshape(coeffs_sp.shape) - coeffs_sp,
            outpath=join(outdir, '1step-sp.png'))
        coeffs_sp_quarter = deepcopy(coeffs_sp)
        coeffs_sp_quarter[(coeffs_sp.shape[0] // 2):,
                          (coeffs_sp.shape[1] // 2):] = 0
        recon_sp = idct(
            idct(coeffs_sp, type=2, norm='ortho').T,
            type=2, norm='ortho').T
        recon_sp_quarter = idct(
            idct(coeffs_sp_quarter, type=2, norm='ortho').T,
            type=2, norm='ortho').T
        cv2.imwrite(join(outdir, 'recon_scipy.png'), recon_sp) # TODO: switch to xm.io.img
        cv2.imwrite(join(outdir, 'recon_scipy_quarter.png'), recon_sp_quarter) # TODO: switch to xm.io.img
        print("(Ours 1-Step Quarter Recon. vs. SciPy) Max. difference: %e" %
              np.abs(recon_sp_quarter.ravel() - recon_1step_quarter).max())

    elif test_id == 'dft_1d_bases':
        signal = np.random.randint(0, 255, 10)
        n = len(signal)
        # Transform by my matrix
        dft_mat = dft_1d_bases(n)
        coeffs = dft_mat.dot(signal)
        # Transform by NumPy
        coeffs_np = np.fft.fft(signal, norm='ortho')
        print("Max. magnitude difference: %e"
              % np.abs(coeffs - coeffs_np).max())

    elif test_id == 'dft_2d_freq':
        h, w = 4, 8
        freq_h, freq_w = dft_2d_freq(h, w)
        print("Along height:")
        print(freq_h)
        print("Along width:")
        print(freq_w)

    elif test_id == 'dft_cameraman':
        from os.path import join
        from . import os as xm_os
        from .vis.matrix import matrix_as_heatmap_complex
        cv2 = preset_import('cv2', assert_success=True)
        outdir = join(const.Dir.tmp, test_id)
        xm_os.makedirs(outdir, rm_if_exists=True)
        im = cv2.imread(const.Path.cameraman, cv2.IMREAD_GRAYSCALE) # TODO: switch to xm.io.img
        im = cv2.resize(im, (64, 64))
        cv2.imwrite(join(outdir, 'orig.png'), im) # TODO: switch to xm.io.img
        # My two-step DFT
        dft_h_mat, dft_w_mat = dft_2d_bases(*im.shape)
        coeffs_2step = dft_h_mat.dot(im).dot(dft_w_mat)
        recon_2step = dft_h_mat.conj().T.dot(
            coeffs_2step).dot(dft_w_mat.conj().T)
        assert np.allclose(np.imag(recon_2step), 0)
        recon_2step = np.real(recon_2step)
        cv2.imwrite(join(outdir, 'recon_2step.png'),
                    recon_2step.astype(im.dtype)) # TODO: switch to xm.io.img
        # NumPy DFT
        coeffs_np = np.fft.fft2(im, norm='ortho')
        recon_np = np.fft.ifft2(coeffs_np, norm='ortho')
        assert np.allclose(np.imag(recon_np), 0)
        recon_np = np.real(recon_np)
        cv2.imwrite(join(outdir, 'recon_np.png'), recon_np.astype(im.dtype)) # TODO: switch to xm.io.img
        # My one-step DFT
        dft_mat = dft_2d_bases_vec(*im.shape)
        coeffs_1step = dft_mat.dot(im.ravel())
        recon_1step = dft_mat.conj().T.dot(coeffs_1step).reshape(im.shape)
        assert np.allclose(np.imag(recon_1step), 0)
        recon_1step = np.real(recon_1step)
        cv2.imwrite(join(outdir, 'recon_1step.png'),
                    recon_1step.astype(im.dtype)) # TODO: switch to xm.io.img
        # Compare coefficients
        matrix_as_heatmap_complex(
            coeffs_1step.reshape(im.shape) - coeffs_np,
            outpath=join(outdir, '1step-np.png'))
        matrix_as_heatmap_complex(
            coeffs_2step - coeffs_1step.reshape(im.shape),
            outpath=join(outdir, '2step-1step.png'))
        matrix_as_heatmap_complex(
            coeffs_2step - coeffs_np, outpath=join(outdir, '2step-np.png'))
        # Quant.
        print("(NumPy vs. Ours Two-Step)\tRecon.\tMax. mag. diff.:\t%e" %
              np.abs(recon_2step - recon_np).max())
        print("(NumPy vs. Ours One-Step)\tRecon.\tMax. mag. diff.:\t%e" %
              np.abs(recon_1step - recon_np).max())

    elif test_id == 'sh_bases_real':
        from os.path import join
        from .vis.matrix import matrix_as_heatmap
        ls = [1, 2, 3, 4]
        n_steps_theta = 64
        for l in ls:
            print("l = %d" % l)
            # Generata harmonics
            ymat, weights = sh_bases_real(
                l, n_steps_theta, _check_orthonormality=False)
            # Black background with white signal
            coeffs_gt = np.random.random(ymat.shape[0])
            sph_func_1d = None
            for ci, c in enumerate(coeffs_gt):
                y_lm = ymat[ci, :]
                if sph_func_1d is None:
                    sph_func_1d = c * y_lm
                else:
                    sph_func_1d += c * y_lm
            sph_func = sph_func_1d.reshape((n_steps_theta, 2 * n_steps_theta))
            sph_func_ravel = sph_func.ravel()
            assert (sph_func_1d == sph_func_ravel).all()
            tmp_dir = const.Dir.tmp
            matrix_as_heatmap(sph_func, outpath=join(tmp_dir, 'sph_orig.png'))
            # Analysis
            coeffs = ymat.dot(np.multiply(weights, sph_func_ravel))
            print("\tGT")
            print(coeffs_gt)
            print("\tRecon")
            print(coeffs)
            # Synthesis
            sph_func_1d_recon = ymat.T.dot(coeffs)
            sph_func_recon = sph_func_1d_recon.reshape(sph_func.shape)
            print("Max. magnitude difference: %e"
                  % np.abs(sph_func_1d - sph_func_1d_recon).max())
            matrix_as_heatmap(sph_func_recon, outpath=join(
                tmp_dir, 'sph_recon_l%03d.png' % l))

    else:
        raise NotImplementedError("Unit tests for %s" % test_id)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('test_id', type=str, help="function to test")
    args = parser.parse_args()

    main(args.test_id)
