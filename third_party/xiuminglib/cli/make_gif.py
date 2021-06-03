"""A commandline tool to make annotated GIF from image-text pairs.

Xiuming Zhang, MIT CSAIL.

July 2017.
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import exists, join, abspath, dirname, basename
from shutil import rmtree
from subprocess import call
from random import choice
from string import ascii_letters, digits
import cv2

from xiuminglib import log
logger, thisfile = log.create_logger(abspath(__file__))
logger.name = thisfile


# Parse variables
parser = ArgumentParser(description="Make annotated GIF from image-text pairs")
parser.add_argument('input', metavar='i', type=str, nargs='+',
                    help="input image-text pairs, e.g., im.png,'foo bar' or im.png")
parser.add_argument('outpath', metavar='o', type=str, help="output GIF")
parser.add_argument('--cropbox', metavar='b', type=str, default='0,0,-1,-1',
                    help=("top left corner, height and width of the cropping rectangle; "
                          "use -1 for \"to the end\"; default: 0,0,-1,-1 (no cropping)"))
parser.add_argument('--delay', metavar='d', type=int, default=200, help="delay parameter; default: 200")
parser.add_argument('--width', metavar='w', type=int, default=1080, help="output GIF width; default: 1080")
parser.add_argument('--fontscale', metavar='s', type=int, default=4, help="font scale; default: 4")
parser.add_argument('--fontthick', metavar='t', type=int, default=5, help="font thickness; default: 5")
parser.add_argument('--fontbgr', metavar='c', type=str, default='0,0,255', help="font BGR; default: 0,0,255 (red)")
parser.add_argument('--anchor', metavar='a', type=str, default='50,50',
                    help="bottom left corner of text box; default: 50,50")
args = parser.parse_args()
pairs = args.input
outpath = abspath(args.outpath)
cropbox = tuple([int(x) for x in args.cropbox.split(',')])
gif_delay = args.delay
gif_width = args.width
font_scale = args.fontscale
font_thick = args.fontthick
font_bgr = tuple([int(x) for x in args.fontbgr.split(',')])
bottom_left_corner = tuple([int(x) for x in args.anchor.split(',')])

# Make directory
outdir = dirname(outpath)
if not exists(outdir):
    makedirs(outdir)

tmpdir = join(outdir, 'tmp_make_gif_' + ''.join(
    [choice(ascii_letters + digits) for n in range(10)]))
if not exists(tmpdir):
    makedirs(tmpdir)

for impath_text in pairs:
    impath_text = impath_text.split(',')
    impath = impath_text[0]

    # Crop, if asked to, and resize
    im = cv2.imread(impath)
    assert im is not None, "%s not found" % impath
    if cropbox != (0, 0, -1, -1):
        y_min, x_min, h, w = cropbox
        if h == -1:
            h = im.shape[0] - y_min
        if w == -1:
            w = im.shape[1] - x_min
        im = im[y_min:(y_min + h), x_min:(x_min + w), ...]
    im = cv2.resize(im, (gif_width, int(im.shape[0] * gif_width / im.shape[1])))

    # Put text
    if len(impath_text) > 1 and impath_text[1] != '':
        text = impath_text[1]
        cv2.putText(im, text, bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_bgr, font_thick)

    # Write new image
    tmppath = join(tmpdir, basename(impath))
    cv2.imwrite(tmppath, im)

# Make GIF
call(['convert', '-delay', str(gif_delay), '-loop', '0',
      join(tmpdir, '*'), outpath])

# Clean up
rmtree(tmpdir)

logger.info("Generated %s", outpath)
