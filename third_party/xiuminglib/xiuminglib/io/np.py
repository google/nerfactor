from os.path import dirname
import numpy as np

from ..os import open_file, exists_isdir, makedirs

from ..log import get_logger
logger = get_logger()


def read_or_write(data_f, fallback=None):
    """Loads the data file if it exists. Otherwise, if fallback is provided,
    call fallback and save its return to disk.

    Args:
        data_f (str): Path to the data file, whose extension will be used for
            deciding how to load the data.
        fallback (function, optional): Fallback function used if data file
            doesn't exist. Its return will be saved to ``data_f`` for future
            loadings. It should not take arguments, but if yours requires taking
            arguments, just wrap yours with::

                fallback=lambda: your_fancy_func(var0, var1)

    Returns:
        Data loaded if ``data_f`` exists; otherwise, ``fallback``'s return
        (``None`` if no fallback).

    Writes
        - Return by the fallback, if provided.
    """
    # Decide data file type
    ext = data_f.split('.')[-1].lower()

    def load_func(path):
        with open_file(path, 'rb') as h:
            data = np.load(h)
        return data

    def save_func(data, path):
        if ext == 'npy':
            save = np.save
        elif ext == 'npz':
            save = np.savez
        else:
            raise NotImplementedError(ext)
        with open_file(path, 'wb') as h:
            save(h, data)

    # Load or call fallback
    if exists_isdir(data_f)[0]:
        data = load_func(data_f)
        msg = "Loaded: "
    else:
        msg = "File doesn't exist "
        if fallback is None:
            data = None
            msg += "(fallback not provided): "
        else:
            data = fallback()
            out_dir = dirname(data_f)
            makedirs(out_dir)
            save_func(data, data_f)
            msg += "(fallback provided); fallback return now saved to: "
    msg += data_f

    logger.info(msg)
    return data
