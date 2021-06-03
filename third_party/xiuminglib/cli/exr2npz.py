"""A commandline tool to convert an EXR image to a .npz dictionary.

Useful when you have problems installing OpenEXR for Python 3. This uses
Python 2 and writes to disk, so that you can later load it with Python 3.

Xiuming Zhang, MIT CSAIL.

Feburary 2018.
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import exists, abspath, dirname
import numpy as np
import OpenEXR
import Imath


# Parse variables
parser = ArgumentParser(
    description="Load OpenEXR image as dictionary of NumPy arrays")
parser.add_argument('input', metavar='i', type=str,
                    help="input .exr file")
parser.add_argument('outpath', metavar='o', type=str,
                    help="output .npz file")
args = parser.parse_args()

inpath = args.input
outpath = abspath(args.outpath)

if not outpath.endswith('.npz'):
    outpath += '.npz'

# Make directory
outdir = dirname(outpath)
if not exists(outdir):
    makedirs(outdir)

f = OpenEXR.InputFile(inpath)
pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
data_win = f.header()['dataWindow']
win_size = (data_win.max.y - data_win.min.y + 1,
            data_win.max.x - data_win.min.x + 1)

imgs = {}
for c in f.header()['channels']:
    arr = np.fromstring(f.channel(c, pix_type), dtype=np.float32)
    imgs[c] = arr.reshape(win_size)

np.savez(outpath, **imgs)
