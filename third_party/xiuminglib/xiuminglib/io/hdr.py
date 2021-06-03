from os.path import dirname
import numpy as np

from ..imprt import preset_import
from ..os import makedirs, open_file

from ..log import get_logger
logger = get_logger()


def read(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    cv2 = preset_import('cv2', assert_success=True)

    with open_file(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    logger.debug("HDR map read from:\n\t%s", path)

    return rgb


def write(rgb, outpath):
    r"""Writes a ``float32`` array as an HDR map to disk.

    Args:
        rgb (numpy.ndarray): ``float32`` RGB array.
        outpath (str): Output path.

    Writes
        - The resultant HDR map.
    """
    cv2 = preset_import('cv2', assert_success=True)
    assert rgb.dtype == np.float32, "Input must be float32"

    makedirs(dirname(outpath))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(outpath, bgr)

    assert success, "Writing HDR failed"
