from os.path import dirname
import numpy as np
from PIL import Image, ExifTags

from ..log import get_logger
logger = get_logger()

from ..os import makedirs, open_file


def load(*args, **kwargs):
    """Alias for :func:`read`, mostly for backward compatibility.

    TODO: remove
    """
    return read(*args, **kwargs)


def write_img(*args, **kwargs):
    """Alias for :func:`write_uint`, mostly for backward compatibility.

    TODO: remove
    """
    return write_uint(*args, **kwargs)


def write_arr(*args, **kwargs):
    """Alias for :func:`write_float`, mostly for backward compatibility.

    TODO: remove
    """
    return write_float(*args, **kwargs)


def read(path, auto_rotate=False):
    """Reads an image from disk.

    Args:
        path (str): Path to the image file. Supported formats: whatever Pillow
            supports.
        auto_rotate (bool, optional): Whether to auto-rotate the read image
            array according to its EXIF orientation, if any.

    Returns:
        numpy.ndarray: Loaded image.
    """
    # EXR and HDR have dedicated loading functions
    if path.endswith('.exr'):
        raise ValueError("Use the dedicated `io.exr.read()` for .exr")
    elif path.endswith('.hdr'):
        raise ValueError("Use the dedicated `io.hdr.read()` for .hdr")

    # Whatever supported by Pillow
    with open_file(path, 'rb') as h:
        img = Image.open(h)
        img.load()

    # Handles the EXIF orientation flag
    if auto_rotate and hasattr(img, 'getexif'):
        exif = img.getexif()
    else:
        exif = None
    if exif is not None and exif:
        exif = dict(exif)
        orientation_key = None
        for orientation_key, _ in ExifTags.TAGS.items():
            if ExifTags.TAGS[orientation_key] == 'Orientation':
                break
        assert orientation_key is not None, "Orientation tag not found"
        orientation = exif[orientation_key]
        if orientation in (1, '1'):
            pass
        elif orientation in (2, '2'):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation in (3, '3'):
            img = img.transpose(Image.ROTATE_180)
        elif orientation in (4, '4'):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation in (5, '5'):
            img = img.transpose(
                Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation in (6, '6'):
            img = img.transpose(Image.ROTATE_270)
        elif orientation in (7, '7'):
            img = img.transpose(
                Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation in (8, '8'):
            img = img.transpose(Image.ROTATE_90)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    img = np.array(img)

    logger.debug("Image read from:\n\t%s", path)

    return img


def write_uint(arr_uint, outpath):
    r"""Writes an ``uint`` array as an image to disk.

    Args:
        arr_uint (numpy.ndarray): A ``uint`` array.
        outpath (str): Output path.

    Writes
        - The resultant image.
    """
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)

    # Write to disk
    makedirs(dirname(outpath))
    with open_file(outpath, 'wb') as h:
        img.save(h)

    logger.debug("Image written to:\n\t%s", outpath)


def write_float(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes a ``float`` array as an image to disk.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            logger.debug("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            logger.debug("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_uint(img_arr, outpath)

    return img_arr
