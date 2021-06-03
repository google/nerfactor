from os.path import join, dirname
import numpy as np

from .text import put_text
from .. import const
from ..os import makedirs
from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


def make_video(
        imgs, fps=24, outpath=None, method='matplotlib', dpi=96, bitrate=-1):
    """Writes a list of images into a grayscale or color video.

    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        method (str, optional): Method to use: ``'matplotlib'``, ``'opencv'``,
            ``'video_api'``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate in kilobits per second when using
            ``matplotlib``; reasonable values include 7200.

    Writes
        - A video of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_video.mp4')
    makedirs(dirname(outpath))

    assert imgs, "Frame list is empty"
    for frame in imgs:
        assert np.issubdtype(frame.dtype, np.unsignedinteger), \
            "Image type must be unsigned integer"

    h, w = imgs[0].shape[:2]
    for frame in imgs[1:]:
        assert frame.shape[:2] == (h, w), \
            "All frames must have the same shape"

    if method == 'matplotlib':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg'] # may require you to specify path
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)
        # If obscure error like "ValueError: Invalid file object: <_io.Buff..."
        # occurs, consider upgrading matplotlib so that it prints out the real,
        # underlying ffmpeg error

        plt.close('all')

    elif method == 'opencv':
        cv2 = preset_import('cv2', assert_success=True)

        # TODO: debug codecs (see http://www.fourcc.org/codecs.php)
        if outpath.endswith('.mp4'):
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # fourcc = 0x00000021
        elif outpath.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise NotImplementedError("Video type of\n\t%s" % outpath)

        vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

        for frame in imgs:
            if frame.ndim == 3:
                frame = frame[:, :, ::-1] # cv2 uses BGR
            vw.write(frame)

        vw.release()

    elif method == 'video_api':
        video_api = preset_import('video_api', assert_success=True)

        assert outpath.endswith('.webm'), "`video_api` requires .webm"

        with video_api.write(outpath, fps=fps) as h:
            for frame in imgs:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                #frame = frame.astype(np.ubyte)
                h.add_frame(frame)

    else:
        raise ValueError(method)

    logger.debug("Images written as a video to:\n%s", outpath)


def make_comparison_video(
        imgs1, imgs2, bar_width=4, bar_color=(1, 0, 0), sweep_vertically=False,
        sweeps=1, label1='', label2='', font_size=None, font_ttf=None,
        label1_top_left_xy=None, label2_top_left_xy=None, **make_video_kwargs):
    """Writes two lists of images into a comparison video that toggles between
    two videos with a sweeping bar.

    Args:
        imgs? (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        bar_width (int, optional): Width of the sweeping bar.
        bar_color (tuple(float), optional): Bar and label RGB, normalized to
            :math:`[0,1]`. Defaults to red.
        sweep_vertically (bool, optional): Whether to sweep vertically or
            horizontally.
        sweeps (int, optional): Number of sweeps.
        label? (str, optional): Label for each video.
        font_size (int, optional): Font size.
        font_ttf (str, optional): Path to the .ttf font file. Defaults to Arial.
        label?_top_left_xy (tuple(int), optional): The XY coordinate of the
            label's top left corner.
        make_video_kwargs (dict, optional): Keyword arguments for
            :func:`make_video`.

    Writes
        - A comparison video.
    """
    # Bar is perpendicular to sweep-along
    sweep_along = 0 if sweep_vertically else 1
    bar_along = 1 if sweep_vertically else 0

    # Number of frames
    n_frames = len(imgs1)
    assert n_frames == len(imgs2), \
        "Videos to be compared have different numbers of frames"

    img_shape = imgs1[0].shape

    # Bar color according to image dtype
    img_dtype = imgs1[0].dtype
    bar_color = np.array(bar_color, dtype=img_dtype)
    if np.issubdtype(img_dtype, np.integer):
        bar_color *= np.iinfo(img_dtype).max

    # Map from frame index to bar location, considering possibly multiple trips
    bar_locs = []
    for i in range(sweeps):
        ind = np.arange(0, img_shape[sweep_along])
        if i % 2 == 1: # reverse every other trip
            ind = ind[::-1]
        bar_locs.append(ind)
    bar_locs = np.hstack(bar_locs) # all possible locations
    ind = np.linspace(0, len(bar_locs) - 1, num=n_frames, endpoint=True)
    bar_locs = [bar_locs[int(x)] for x in ind] # uniformly sampled

    # Label locations
    if label1_top_left_xy is None:
        # Label 1 at top left corner
        label1_top_left_xy = (int(0.1 * img_shape[1]), int(0.05 * img_shape[0]))
    if label2_top_left_xy is None:
        if sweep_vertically:
            # Label 2 at bottom left corner
            label2_top_left_xy = (
                int(0.1 * img_shape[1]), int(0.75 * img_shape[0]))
        else:
            # Label 2 at top right corner
            label2_top_left_xy = (
                int(0.7 * img_shape[1]), int(0.05 * img_shape[0]))

    frames = []
    for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        assert img1.shape == img_shape, f"`imgs1[{i}]` has a differnet shape"
        assert img2.shape == img_shape, f"`imgs2[{i}]` has a differnet shape"
        assert img1.dtype == img_dtype, f"`imgs1[{i}]` has a differnet dtype"
        assert img2.dtype == img_dtype, f"`imgs2[{i}]` has a differnet dtype"

        # Label the two images
        img1 = put_text(
            img1, label1, label_top_left_xy=label1_top_left_xy,
            font_size=font_size, font_color=bar_color, font_ttf=font_ttf)
        img2 = put_text(
            img2, label2, label_top_left_xy=label2_top_left_xy,
            font_size=font_size, font_color=bar_color, font_ttf=font_ttf)

        # Bar start and end
        bar_loc = bar_locs[i]
        bar_width_half = bar_width // 2
        bar_start = max(0, bar_loc - bar_width_half)
        bar_end = min(bar_loc + bar_width_half, img_shape[sweep_along])

        # Up to bar start, we show Image 1; bar end onwards, Image 2
        img1 = np.take(img1, range(bar_start), axis=sweep_along)
        img2 = np.take(
            img2, range(bar_end, img_shape[sweep_along]), axis=sweep_along)

        # Between the two images, we show the bar
        actual_bar_width = img_shape[
            sweep_along] - img1.shape[sweep_along] - img2.shape[sweep_along]
        reps = [1, 1, 1]
        reps[sweep_along] = actual_bar_width
        reps[bar_along] = img_shape[bar_along]
        bar_img = np.tile(bar_color, reps)

        frame = np.concatenate((img1, bar_img, img2), axis=sweep_along)
        frames.append(frame)

    make_video(frames, **make_video_kwargs)
