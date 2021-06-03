from os.path import join, dirname
import numpy as np
from PIL import Image

from ..log import get_logger
logger = get_logger()

from .. import const
from ..io import img as imgio
from ..os import makedirs, open_file


def make_anim(imgs, duration=1, outpath=None):
    r"""Writes a list of images into an animation.

    In most cases, we need to label each image, for which you can use
    :func:`vis.text.put_text`.

    Args:
        imgs (list(numpy.ndarray or str)): An image is either a path or an
            array (mixing ok, but arrays will need to be written to a temporary
            directory). If array, should be of type ``uint`` and of shape H-by-W
            (grayscale) or H-by-W-by-3 (RGB).
        duration (float, optional): Duration of each frame in seconds.
        outpath (str, optional): Where to write the output to (a .apng or .gif
            file). ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_anim.gif')``.

    Writes
        - An animation of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_anim.gif')
    if not outpath.endswith(('.apng', '.gif')):
        outpath += '.gif'
    makedirs(dirname(outpath))

    imgs_loaded = []
    for img in imgs:
        if isinstance(img, str):
            # Path
            img = imgio.load(img)
            imgs_loaded.append(img)
        elif isinstance(img, np.ndarray):
            # Array
            assert np.issubdtype(img.dtype, np.unsignedinteger), \
                "If image is provided as an array, it has to be `uint`"
            if (img.ndim == 3 and img.shape[2] == 1) or img.ndim == 2:
                img = np.dstack([img] * 3)
            img = Image.fromarray(img)
            imgs_loaded.append(img)
        else:
            raise TypeError(type(img))

    duration = duration * 1000 # because in ms

    with open_file(outpath, 'wb') as h:
        imgs_loaded[0].save(
            h, save_all=True, append_images=imgs_loaded[1:],
            duration=duration, loop=0)

    logger.debug("Images written as an animation to:\n\t%s", outpath)
