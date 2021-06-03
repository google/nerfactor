"""A commandline tool to convert a .py script to a .ipynb file.

Xiuming Zhang, MIT CSAIL.

April 2019.
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import exists, abspath, dirname
from nbformat import v4, write


# Parse variables
parser = ArgumentParser(description="Convert a .py script to a .ipynb file")
parser.add_argument('input', metavar='i', type=str, help="input .py file")
parser.add_argument('outpath', metavar='o', type=str, help="output .ipynb file")
args = parser.parse_args()
inpath = args.input
outpath = abspath(args.outpath)

if not outpath.endswith('.ipynb'):
    outpath += '.ipynb'

# Make directory
outdir = dirname(outpath)
if not exists(outdir):
    makedirs(outdir)

nb = v4.new_notebook()
with open(inpath, 'r') as h:
    code = h.read()
    nb.cells.append(v4.new_code_cell(code))
write(nb, outpath)
