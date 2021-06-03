from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


def read(path):
    """Reads a non-multi-layer OpenEXR image from disk.

    Reading a multi-layer OpenEXR cannot be done with OpenCV and would require
    installing OpenEXR and Imath (see `cli/exr2npz.py`).

    Args:
        path (str): Path to the .exr file.

    Returns:
        numpy.ndarray: Loaded (float) array with RGB channels in order.
    """
    cv2 = preset_import('cv2', assert_success=True)

    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    logger.debug(f"Read {path}")

    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb

    raise NotImplementedError(arr.shape)
