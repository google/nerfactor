from os.path import dirname
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import const
from ..os import makedirs, open_file
from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


def put_text(
        img, text, label_top_left_xy=None, font_size=None, font_color=(1, 0, 0),
        font_ttf=None):
    r"""Puts text on image.

    Args:
        img (numpy.ndarray): Should be of type ``uint`` and of shape H-by-W
            (grayscale) or H-by-W-by-3 (RGB).
        text (str): Text to be written on the image.
        label_top_left_xy (tuple(int), optional): The XY coordinate of the
            label's top left corner.
        font_size (int, optional): Font size.
        font_color (tuple(float), optional): Font RGB, normalized to
            :math:`[0,1]`. Defaults to red.
        font_ttf (str, optional): Path to the .ttf font file. Defaults to Arial.

    Returns:
        numpy.ndarray: The modified image with text.
    """
    assert np.issubdtype(img.dtype, np.integer) and (
        not np.issubdtype(img.dtype, np.signedinteger)), \
        "Input image must be `uint` (i.e., an actual image)"

    if font_size is None:
        font_size = int(0.1 * img.shape[0])

    # Font
    if font_ttf is None:
        font = ImageFont.truetype(const.Path.open_sans_regular, font_size)
    else:
        with open_file(font_ttf, 'rb') as h:
            font_bytes = BytesIO(h.read())
        font = ImageFont.truetype(font_bytes, font_size)

    if label_top_left_xy is None:
        label_top_left_xy = (int(0.1 * img.shape[1]), int(0.05 * img.shape[0]))

    dtype_max = np.iinfo(img.dtype).max
    color = tuple(int(x * dtype_max) for x in font_color)

    img = Image.fromarray(img)
    img = img.convert('RGB') # to avoid errors due to RGB text on grayscale
    ImageDraw.Draw(img).text(label_top_left_xy, text, fill=color, font=font)
    img = np.array(img)

    return img


def text_as_image(
        text, imsize=256, thickness=2, dtype='uint8', outpath=None,
        quiet=False):
    """Rasterizes a text string into an image.

    The text will be drawn in white to the center of a black canvas.
    Text size gets automatically figured out based on the provided
    thickness and image size.

    Args:
        text (str): Text to be drawn.
        imsize (float or tuple(float), optional): Output image height and width.
        thickness (float, optional): Text thickness.
        dtype (str, optional): Image type.
        outpath (str, optional): Where to dump the result to. ``None``
            means returning instead of writing it.
        quiet (bool, optional): Whether to refrain from logging.
            Effective only when ``outpath`` is not ``None``.

    Returns or Writes
        - An image of the text.
    """
    cv2 = preset_import('cv2', assert_success=True)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)
    assert isinstance(imsize, tuple), \
        "`imsize` must be an int or a 2-tuple of ints"

    # Unimportant constants not exposed to the user
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    base_bgr = (0, 0, 0) # black
    text_bgr = (1, 1, 1) # white

    # Base canvas
    dtype_max = np.iinfo(dtype).max
    im = np.tile(base_bgr, imsize + (1,)).astype(dtype) * dtype_max

    # Figure out the correct font scale
    font_scale = 1 / 128 # real small
    while True:
        (text_width, text_height), bl_y = cv2.getTextSize(
            text, font_face, font_scale, thickness)
        if bl_y + text_height >= imsize[0] or text_width >= imsize[1]:
            # Undo the destroying step before breaking
            font_scale /= 2
            (text_width, text_height), bl_y = cv2.getTextSize(
                text, font_face, font_scale, thickness)
            break
        font_scale *= 2

    # Such that the text is at the center
    bottom_left_corner = (
        (imsize[1] - text_width) // 2,
        (imsize[0] - text_height) // 2 + text_height)
    cv2.putText(
        im, text, bottom_left_corner, font_face, font_scale,
        [x * dtype_max for x in text_bgr], thickness)

    if outpath is None:
        return im

    # Write
    outdir = dirname(outpath)
    makedirs(outdir)
    cv2.imwrite(outpath, im)

    if not quiet:
        logger.info("Text rasterized into image to:\n%s", outpath)
