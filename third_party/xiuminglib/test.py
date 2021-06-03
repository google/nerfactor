from os.path import join
import numpy as np

from absl import app

try:
    from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
except ModuleNotFoundError:
    import xiuminglib as xm


def main(_):
    json_path = join(xm.const.Dir.tmp, 'transforms_train.json')
    data = xm.io.json.load(json_path)
    xm.io.json.write(data, json_path[:-len('.json')] + '_repro.json')

    return

    dtype = 'uint8'
    n_ch = 3
    ims = 256
    ssim = xm.metric.SSIM(dtype)
    psnr = xm.metric.PSNR(dtype)
    lpips = xm.metric.LPIPS(dtype)
    dtype_max = np.iinfo(dtype).max
    im1 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    im2 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    print("SSIM", ssim(im1, im2))
    print("MS-SSIM", ssim(im1, im2, multiscale=True))
    print("PSNR", psnr(im1, im2))
    print("LPIPS", lpips(im1, im2))

    return

    plot = xm.vis.plot.Plot()
    y = np.random.uniform(size=(16, 4))
    plot.bar(y, labels=('A', 'B', 'C', 'D'))
    xyz = np.random.uniform(size=(128, 3))
    plot.scatter3d(xyz)

    return

    logger = xm.log.get_logger()
    logger.info("This is INFO")
    logger.warning("This is WARNING")
    logger.error("This is ERROR")

    return

    im_linear = np.random.rand(256, 256, 3)
    im_srgb = xm.img.linear2srgb(im_linear)
    im_srgb_linear = xm.img.srgb2linear(im_srgb)
    print(np.abs(im_linear - im_srgb_linear).max())

    return


if __name__ == '__main__':
    app.run(main)
