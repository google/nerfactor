from os.path import join, dirname
import numpy as np

from .. import const, os as xm_os
from .general import _savefig
from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


def scatter_on_img(pts, im, size=2, bgr=(0, 0, 255), outpath=None):
    r"""Plots scatter on top of an image or just a white canvas, if you are
    being creative by feeding in just a white image.

    Args:
        pts (array_like): Pixel coordinates of the scatter point(s), of length 2
            for just one point or shape N-by-2 for multiple points.
            Convention:

            .. code-block:: none

                +-----------> dim1
                |
                |
                |
                v dim0

        im (numpy.ndarray): Image to scatter on. H-by-W (grayscale) or
            H-by-W-by-3 (RGB) arrays of ``unint`` type.
        size (float or array_like(float), optional): Size(s) of scatter
            points. If *array_like*, must be of length N.
        bgr (tuple or array_like(tuple), optional): BGR color(s) of scatter
            points. Each element :math:`\in [0, 255]`. If *array_like*, must
            be of shape N-by-3.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means ``os.path.join(const.Dir.tmp,
            'scatter_on_img.png')``.

    Writes
        - The scatter plot overlaid over the image.
    """
    cv2 = preset_import('cv2', assert_success=True)

    if outpath is None:
        outpath = join(const.Dir.tmp, 'scatter_on_img.png')

    thickness = -1 # for filled circles

    # Standardize inputs
    if im.ndim == 2: # grayscale
        im = np.dstack((im, im, im)) # to BGR
    pts = np.array(pts)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    n_pts = pts.shape[0]

    if im.dtype != 'uint8' and im.dtype != 'uint16':
        logger.warning("Input image type may cause obscure cv2 errors")

    if isinstance(size, int):
        size = np.array([size] * n_pts)
    else:
        size = np.array(size)

    bgr = np.array(bgr)
    if bgr.ndim == 1:
        bgr = np.tile(bgr, (n_pts, 1))

    # FIXME: necessary, probably due to OpenCV bugs?
    im = im.copy()

    # Put on scatter points
    for i in range(pts.shape[0]):
        xy = tuple(pts[i, ::-1].astype(int))
        color = (int(bgr[i, 0]), int(bgr[i, 1]), int(bgr[i, 2]))
        cv2.circle(im, xy, size[i], color, thickness)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Write to disk
    cv2.imwrite(outpath, im) # TODO: switch to xm.io.img


def uv_on_texmap(uvs, texmap, ft=None, outpath=None, max_n_lines=None,
                 dotsize=4, dotcolor='r', linewidth=1, linecolor='b'):
    """Visualizes which points on texture map the vertices map to.

    Args:
        uvs (numpy.ndarray): N-by-2 array of UV coordinates. See
            :func:`xiuminglib.blender.object.smart_uv_unwrap` for the UV
            coordinate convention.
        texmap (numpy.ndarray or str): Loaded texture map or its path. If
            *numpy.ndarray*, can be H-by-W (grayscale) or H-by-W-by-3 (color).
        ft (list(list(int)), optional): Texture faces used to connect the
            UV points. Values start from 1, e.g., ``'[[1, 2, 3], [],
            [2, 3, 4, 5], ...]'``.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'uv_on_texmap.png')``.
        max_n_lines (int, optional): Plotting a huge number of lines can be
            slow, so set this to uniformly sample a subset to plot. Useless if
            ``ft`` is ``None``.
        dotsize (int or list(int), optional): Size(s) of the UV dots.
        dotcolor (str or list(str), optional): Their color(s).
        linewidth (float, optional): Width of the lines connecting the dots.
        linecolor (str, optional): Their color.

    Writes
        - An image of where the vertices map to on the texture map.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if outpath is None:
        outpath = join(const.Dir.tmp, 'uv_on_texmap.png')

    # Preprocess input
    if isinstance(texmap, str):
        cv2 = preset_import('cv2', assert_success=True)
        texmap = cv2.imread( # TODO: switch to xm.io.img
            texmap, cv2.IMREAD_UNCHANGED)[:, :, ::-1] # made RGB
    if len(texmap.shape) == 2:
        add_colorbar = True # for grayscale
    elif len(texmap.shape) == 3:
        add_colorbar = False # for color texture maps
    else:
        raise ValueError(
            ("texmap must be either H-by-W (grayscale) or H-by-W-by-3 "
             "(color), or a path to such images"))

    dpi = 96 # assumed
    h, w = texmap.shape[:2]
    w_in, h_in = w / dpi, h / dpi
    fig = plt.figure(figsize=(w_in, h_in))

    u, v = uvs[:, 0], uvs[:, 1]
    # ^ v
    # |
    # +---> u
    x, y = u * w, (1 - v) * h
    #   +---> x
    #   |
    #   v y

    # UV dots
    ax = fig.gca()
    ax.set_xlim([min(0, min(x)), max(w, max(x))])
    ax.set_ylim([max(h, max(y)), min(0, min(y))])
    im = ax.imshow(texmap, cmap='gray')
    ax.scatter(x, y, c=dotcolor, s=dotsize, zorder=2)
    ax.set_aspect('equal')

    # Connect these dots
    if ft is not None:
        lines = []
        for vert_id in [x for x in ft if x]: # non-empty ones
            assert min(vert_id) >= 1, "Indices in ft are 1-indexed"
            # For each face
            ind = [i - 1 for i in vert_id]
            n_verts = len(ind)
            for i in range(n_verts):
                lines.append([
                    (x[ind[i]], y[ind[i]]),
                    (x[ind[(i + 1) % n_verts]], y[ind[(i + 1) % n_verts]])
                ]) # line start and end
        if max_n_lines is not None:
            lines = [lines[i] for i in np.linspace(
                0, len(lines) - 1, num=max_n_lines, dtype=int)]
        line_collection = LineCollection(
            lines, linewidths=linewidth, colors=linecolor, zorder=1)
        ax.add_collection(line_collection)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Colorbar
    if add_colorbar:
        # Create an axes on the right side of ax. The width of cax will be 2%
        # of ax and the padding between cax and ax will be fixed at 0.1 inch.
        cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.2)
        plt.colorbar(im, cax=cax)

    # Save
    contents_only = not add_colorbar
    _savefig(outpath, contents_only=contents_only)

    plt.close('all')
