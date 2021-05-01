import cv2


def read_exr(path):
    bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert bgr is not None, "Loading failed"
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise NotImplementedError(bgr.shape)
    return bgr[:, :, ::-1]
