#!/usr/bin/env python

from os.path import join, exists, basename
from glob import glob
from tqdm import tqdm

import xiuminglib as xm


def main():
    scene = (
        'lego',
        'hotdog',
    )[1] # NOTE
    envmap = (
        'studio',
        'interior',
        '2159',
        '2234',
        '3072',
        '3083',
    )[-1] # NOTE
    fps = 6 # NOTE
    data_root = join( # NOTE
        '/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3',
        f'{scene}_{envmap}', '{mode}_???')
    vis_mp4 = join( # NOTE
        '/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3',
        f'{scene}_{envmap}', '{mode}.mp4')

    for mode in ('train', 'val', 'test'):
        frames = []
        for view_dir in tqdm(
                sorted(glob(data_root.format(mode=mode))), desc=mode):
            view_id = basename(view_dir)
            rgba_png = join(view_dir, 'rgba.png')
            if exists(rgba_png):
                rgba = xm.io.img.load(rgba_png)
                rgb = rgba[:, :, :3]
                rgb = xm.vis.text.put_text(rgb, view_id)
                frames.append(rgb)
        xm.vis.video.make_video(
            frames, outpath=vis_mp4.format(mode=mode), fps=fps)


if __name__ == '__main__':
    main()
