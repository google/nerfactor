"""Decorators that wrap a function.

If the function is defined in the file where you want to use the decorator,
you can decorate the function at define time:

.. code-block:: python

    @decorator
    def somefunc():
        return

If the function is defined somewhere else, do:

.. code-block:: python

    from numpy import mean

    mean = decorator(mean)
"""

from time import time, sleep
from os import makedirs, environ
import os.path
from os.path import join, dirname, getmtime

from .os import _is_cnspath, _no_trailing_slash, cp, rm

from .log import get_logger
logger = get_logger()


def colossus_interface(somefunc):
    """Wraps black-box functions to read from and write to Google Colossus.

    Because it's hard (if possible at all) to figure out which path is
    input, and which is output, when the input function is black-box, this is
    a "best-effort" decorator (see below for warnings).

    This decorator works by looping through all the positional and keyword
    parameters, copying CNS paths that exist prior to ``somefunc`` execuation
    to temporary local locations, running ``somefunc`` and writing its output
    to local locations, and finally copying local paths that get modified by
    ``somefunc`` to their corresponding CNS locations.

    Warning:
        Therefore, if ``somefunc``'s output already exists (e.g., you are
        re-running the function to overwrite the old result), it will be
        copied to local, overwritten by ``somefunc`` locally, and finally
        copied back to CNS. This doesn't lead to wrong behaviors, but is
        inefficient.

    This decorator doesn't depend on Blaze, as it's using the ``fileutil``
    CLI, rather than ``google3.pyglib.gfile``. This is convenient in at least
    two cases:

    - You are too lazy to use Blaze, want to run tests quickly on your local
      machine, but need access to CNS files.
    - Your IO is more complex than what ``with gfile.Open(...) as h:`` can do
      (e.g., a Blender function importing an object from a path), in which
      case you have to copy the CNS file to local ("local" here could also
      mean a Borglet's local).

    This interface generally works with resolved paths (e.g.,
    ``/path/to/file``), but not with wildcard paths (e.g., ``/path/to/???``),
    sicne it's hard (if possible at all) to guess what your function tries to
    do with such wildcard paths.

    Writes
        - Input files copied from Colossus to ``$TMP/``.
        - Output files generated to ``$TMP/``, to be copied to Colossus.
    """
    # $TMP set by Borg or yourself (e.g., with .bashrc)
    tmp_dir = environ.get('TMP', '/tmp/')

    def gen_local_path(cns_path):
        keep_last_n = 3
        cns_path = _no_trailing_slash(cns_path)
        local_basename = '_'.join(cns_path.split('/')[-keep_last_n:])
        # Prefixed by time to avoid dupes
        local_path = join(tmp_dir, '%f_%s' % (time(), local_basename))
        # local_path guaranteed to not end with '/'
        return local_path

    def cp_404ok(src, dst):
        try:
            cp(src, dst)
            logger.debug("\n%s\n\tcopied to\n%s", src, dst)
        except FileNotFoundError:
            logger.warning(("Source doesn't exist yet:\n\t%s\n"
                            "OK if this will be the output"), src)

    def wrapper(*arg, **kwargs):
        t_eps = 0.05 # seconds buffer for super fast somefunc
        # Fetch info. for all CNS paths
        arg_local, kwargs_local = [], {}
        cns2local = {}
        # Positional arguments
        for x in arg:
            if _is_cnspath(x):
                local_path = gen_local_path(x)
                cns2local[x] = local_path
                arg_local.append(local_path)
            else: # intact
                arg_local.append(x)
        # Keyword arguments
        for k, v in kwargs.items():
            if _is_cnspath(v):
                local_path = gen_local_path(v)
                cns2local[v] = local_path
                kwargs_local[k] = local_path
            else: # intact
                kwargs_local[k] = v
        # For reading: copy CNS paths that exist to local
        # TODO: what if some of those paths are not input? Copying them to
        # local is a waste (but harmless)
        for cns_path, local_path in cns2local.items():
            cp_404ok(cns_path, local_path)
        # Run the real function
        t0 = time()
        sleep(t_eps)
        results = somefunc(*arg_local, **kwargs_local)
        # For writing: copy local paths that are just modified and correspond
        # to CNS paths back to CNS
        for cns_path, local_path in cns2local.items():
            if os.path.exists(local_path) and getmtime(local_path) > t0:
                cp_404ok(local_path, cns_path)
        # Free up the space by deleting the temporary files
        for _, local_path in cns2local.items():
            rm(local_path)
        return results

    return wrapper


def timeit(somefunc):
    """Outputs the time a function takes to execute."""
    def wrapper(*arg, **kwargs):
        t0 = time()
        results = somefunc(*arg, **kwargs)
        t = time() - t0
        logger.info("Time elapsed: %f seconds", t)
        return results

    return wrapper


def existok(makedirs_func):
    """Implements the ``exist_ok`` flag in 3.2+, which avoids race conditions,
    where one parallel worker checks the folder doesn't exist and wants to
    create it with another worker doing so faster.
    """
    def wrapper(*args, **kwargs):
        try:
            makedirs_func(*args, **kwargs)
        except OSError as e:
            if e.errno != 17:
                raise
            logger.debug("%s already exists, but that is OK", args[0])

    return wrapper


def main():
    """Unit tests that can also serve as example usage."""
    # timeit
    @timeit
    def findsums(x, y, z):
        sleep(1)
        return x + y, x + z, y + z, x + y + z
    print(findsums(1, 2, 3))

    # existok
    newdir = join(dirname(__file__), 'test')
    makedirs_ = existok(makedirs)
    makedirs_(newdir)
    makedirs_(newdir)


if __name__ == '__main__':
    main()
