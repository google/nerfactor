from os.path import dirname
import json

from ..os import makedirs, open_file

from .. import log
logger = log.get_logger()


def write(dict_, path):
    """Writes a dictionary into a JSON.

    Args:
        dict_ (dict): Data dictionary.
        path (str): Path to the JSON file.

    Writes
        - JSON file.
    """
    outdir = dirname(path)
    makedirs(outdir)

    with open_file(path, 'w') as h:
        json.dump(dict_, h, indent=4, sort_keys=True)


def load(path):
    """Loads a JSON.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Data dictionary.
    """
    with open_file(path, 'r') as h:
        data = json.load(h)
    return data
