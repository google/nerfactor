"""A commandline tool to extract frames from a video.

Xiuming Zhang, MIT CSAIL.

May 2017.
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import exists, join, abspath
from cv2 import imwrite, VideoCapture

from xiuminglib import log
logger, thisfile = log.create_logger(abspath(__file__))
logger.name = thisfile


# Parse variables
parser = ArgumentParser(description="Extract frames from a video file")
parser.add_argument('videopath', metavar='i', type=str, help="input video file")
parser.add_argument('outdir', metavar='o', type=str, help="output directory")
parser.add_argument('--every', metavar='n', type=int, default=1,
                    help="sample one frame every n frame(s) (default: 1)")
parser.add_argument('--outlen', metavar='l', type=int, default=4,
                    help="length of output filenames (default: 4)")
args = parser.parse_args()
videopath = args.videopath
outdir = abspath(args.outdir)
every = args.every
outlen = args.outlen

# Make directory
if not exists(outdir):
    makedirs(outdir)

# Read frames from video
vid = VideoCapture(videopath)
frameidx = 0
frameidx_out = 1
while vid.isOpened():
    success, im = vid.read()
    if not success:
        break
    if frameidx % every == 0:
        outpath = join(outdir, str(frameidx_out).zfill(outlen) + '.png')
        logger.info("Frame %d saved as %s", frameidx, outpath)
        imwrite('%s' % outpath, im)
        frameidx_out += 1
    frameidx += 1
vid.release()
