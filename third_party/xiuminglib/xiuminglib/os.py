import os
from os.path import join, exists, isdir, dirname
from shutil import rmtree, copy2, copytree
from glob import glob

from .log import get_logger
logger = get_logger()

from .imprt import preset_import
from .interact import format_print


def open_file(path, mode):
    """Opens a file.

    Supports Google Colossus if ``gfile`` can be imported.

    Args:
        path (str): Path to open.
        mode (str): ``'r'``, ``'rb'``, ``'w'``, or ``'wb'``.

    Returns:
        File handle that can be used as a context.
    """
    gfile = preset_import('gfile')

    open_func = open if gfile is None else gfile.Open
    handle = open_func(path, mode)

    return handle


def _is_cnspath(path):
    return isinstance(path, str) and path.startswith('/cns/')


def _is_bspath(path):
    return isinstance(path, str) and path.startswith('/bigstore/')


def sortglob(directory, filename='*', ext=None, ext_ignore_case=False):
    """Globs and then sorts filenames, possibly ending with multiple
    extensions, in a directory.

    Supports Google Colossus, by using ``gfile`` (preferred for speed)
    or the ``fileutil`` CLI when Blaze is not used (hence, ``gfile``
    unavailable).

    Args:
        directory (str): Directory to glob, e.g., ``'/path/to/'``.
        filename (str or tuple(str), optional): Filename pattern excluding
            extensions, e.g., ``'img*'``.
        ext (str or tuple(str), optional): Extensions of interest, e.g.,
            ``('png', 'jpg')``. ``None`` means no extension, useful for
            folders or files with no extension.
        ext_ignore_case (bool, optional): Whether to ignore case for
            extensions.

    Returns:
        list(str): Sorted list of files globbed.
    """
    def glob_cns_cli(pattern):
        cmd = 'fileutil ls -d %s' % pattern # -d to avoid recursively
        _, stdout, _ = call(cmd, quiet=True)
        return [x for x in stdout.split('\n') if x != '']

    def glob_bs_cli(pattern):
        cmd = '/google/data/ro/projects/cloud/bigstore/fileutil_bs ls -d %s' \
            % pattern # -d to avoid recursively
        _, stdout, _ = call(cmd, quiet=True)
        return [x for x in stdout.split('\n') if x != '']

    if _is_cnspath(directory):
        # Is a CNS path
        gfile = preset_import('gfile', assert_success=True)
        if gfile is None:
            glob_func = glob_cns_cli
        else:
            glob_func = gfile.Glob
    elif _is_bspath(directory):
        # Is a Bigstore path
        gfile = preset_import('gfile', assert_success=True)
        if gfile is None:
            glob_func = glob_bs_cli
        else:
            glob_func = gfile.Glob
    else:
        # Is just a regular local path
        glob_func = glob

    if ext is None:
        ext = ()
    elif isinstance(ext, str):
        ext = (ext,)
    if isinstance(filename, str):
        filename = (filename,)

    ext_list = []
    for x in ext:
        if not x.startswith('.'):
            x = '.' + x
        if ext_ignore_case:
            ext_list += [x.lower(), x.upper()]
        else:
            ext_list.append(x)

    files = []
    for f in filename:
        if ext_list:
            for e in ext_list:
                files += glob_func(join(directory, f + e))
        else:
            files += glob_func(join(directory, f))

    files_sorted = sorted(files)
    return files_sorted


def exists_isdir(path):
    """Determines whether a path exists, and if so, whether it is a file
    or directory.

    Supports Google Colossus (CNS) paths by using ``gfile`` (preferred for
    speed) or the ``fileutil`` CLI.

    Args:
        path (str): A path.

    Returns:
        tuple:
            - **exists** (*bool*) -- Whether the path exists.
            - **isdir** (*bool*) -- Whether the path is a file or directory.
              ``None`` if the path doesn't exist.
    """
    path = _no_trailing_slash(path)

    # If local path, do the job quickly and return
    if not _is_cnspath(path):
        path_exists = exists(path)
        path_isdir = isdir(path) if path_exists else None
        return path_exists, path_isdir

    gfile = preset_import('gfile', assert_success=True)

    # Using fileutil CLI
    if gfile is None:
        testf, _, _ = call('fileutil test -f %s' % path)
        testd, _, _ = call('fileutil test -d %s' % path)
        if testf == 1 and testd == 1:
            path_exists = False
            path_isdir = None
        elif testf == 1 and testd == 0:
            path_exists = True
            path_isdir = True
        elif testf == 0 and testd == 1:
            path_exists = True
            path_isdir = False
        else:
            raise NotImplementedError("What does this even mean?")

    # Using gfile
    else:
        path_exists = gfile.Exists(path)
        if path_exists:
            path_isdir = gfile.IsDirectory(path)
        else:
            path_isdir = None

    return path_exists, path_isdir


def _no_trailing_slash(path):
    if path.endswith('/'):
        path = path[:-1]
    assert not path.endswith('/'), "path shouldn't end with '//'"
    # Guaranteed to not end with '/', so basename() or dirname()
    # will give the correct results
    return path


def _select_gfs_user(writeto):
    """As whom we perform file operations.

    Useful for operations on a folder whose owner is a Ganpati group (e.g.,
    ``gcam-gpu``).
    """
    gfile = preset_import('gfile', assert_success=True)

    writeto = _no_trailing_slash(writeto)

    writeto_exists, writeto_isdir = exists_isdir(writeto)
    if writeto_exists and writeto_isdir:
        # OK as long as we can write to it
        writeto_folder = writeto
    else:
        # Doesn't exist yet or is a file, so we need to write to its parent
        writeto_folder = dirname(writeto)

    if gfile is None:
        stdout = _call_assert_success(
            'fileutil ls -l -d %s' % writeto_folder, quiet=True)
        assert stdout.count('\n') == 1, \
            "`fileuti ls` results should have one line only"
        owner = stdout.strip().split(' ')[2]
    else:
        owner = gfile.Stat(writeto_folder).owner

    return owner


def cp(src, dst, cns_parallel_copy=10):
    """Copies files, possibly from/to the Google Colossus Filesystem.

    Args:
        src (str): Source file or directory.
        dst (str): Destination file or directory.
        cns_parallel_copy (int): The number of files to be copied in
            parallel. Only effective when copying a directory from/to
            Colossus.
    """
    src = _no_trailing_slash(src)
    dst = _no_trailing_slash(dst)

    srcexists, srcisdir = exists_isdir(src)
    if not srcexists:
        raise FileNotFoundError("Source must exist")

    # When no CNS paths involved, quickly do the job and return
    if not _is_cnspath(src) and not _is_cnspath(dst):
        if srcisdir:
            for x in os.listdir(src):
                s = join(src, x)
                d = join(dst, x)
                if isdir(s):
                    copytree(s, d)
                else:
                    copy2(s, d)
        else:
            copy2(src, dst)
        return

    gfile = preset_import('gfile', assert_success=True)

    if gfile is None:
        cmd = 'fileutil cp -f -colossus_parallel_copy '
        if srcisdir:
            cmd += '-R -parallel_copy=%d %s ' % \
                (cns_parallel_copy, join(src, '*'))
        else:
            cmd += '%s ' % src
        cmd += '%s' % dst
        # Destination directory may be owned by a Ganpati group
        if _is_cnspath(dst):
            cmd += ' --gfs_user %s' % _select_gfs_user(dst)
        _call_assert_success(cmd)

    else:
        with gfile.AsUser(_select_gfs_user(dst)):
            if srcisdir:
                gfile.RecursivelyCopyDir(src, dst, overwrite=True)
            else:
                gfile.Copy(src, dst, overwrite=True)


def rm(path):
    """Removes a file or recursively a directory, with Google Colossus
    compatibility.

    Args:
        path (str)
    """
    if not _is_cnspath(path):
        # Quickly do the job and return
        if exists(path):
            if isdir(path):
                rmtree(path)
            else:
                os.remove(path)
        return

    # OK, a CNS path
    # Use gfile if available
    gfile = preset_import('gfile')
    if gfile is not None:
        gfile.DeleteRecursively(path) # works for file and directory
    else:
        # Falls back to filter CLI
        cmd = 'fileutil rm -R -f %s' % path # works for file and directory
        _call_assert_success(cmd, quiet=True)


def makedirs(directory, rm_if_exists=False):
    """Wraps :func:`os.makedirs` to support removing the directory if it
    alread exists.

    Google Colossus-compatible: it tries to use ``gfile`` first for speed. This
    will fail if Blaze is not used, in which case it then falls back to using
    ``fileutil`` CLI as external process calls.

    Args:
        directory (str)
        rm_if_exists (bool, optional): Whether to remove the directory (and
            its contents) if it already exists.
    """
    def exists_cns_cli(directory):
        cmd = 'fileutil test -d %s' % directory
        retcode, _, _ = call(cmd, quiet=True)
        if retcode == 0:
            return True
        if retcode == 1:
            return False
        raise ValueError(retcode)

    def mkdir_cns_cli(directory):
        cmd = 'fileutil mkdir -p %s' % directory
        _call_assert_success(cmd, quiet=True)

    if _is_cnspath(directory):
        # Is a CNS path
        gfile = preset_import('gfile')
        if gfile is None:
            exists_func = exists_cns_cli
            mkdir_func = mkdir_cns_cli
        else:
            exists_func = gfile.Exists
            mkdir_func = gfile.MakeDirs
    else:
        # Is just a regular local path
        exists_func = exists
        mkdir_func = os.makedirs

    # Do the job
    if exists_func(directory):
        if rm_if_exists:
            rm(directory)
            mkdir_func(directory)
            logger.info("Removed and then remade:\n\t%s", directory)
    else:
        mkdir_func(directory)


def make_exp_dir(directory, param_dict, rm_if_exists=False):
    """Makes an experiment output folder by hashing the experiment parameters.

    Args:
        directory (str): The made folder will be under this.
        param_dict (dict): Dictionary of the parameters identifying the
            experiment. It is sorted by its keys, so different orders lead to
            the same hash.
        rm_if_exists (bool, optional): Whether to remove the experiment folder
            if it already exists.

    Writes
        - The experiment parameters in ``<directory>/<hash>/param.json``.

    Returns:
        str: The experiment output folder just made.
    """
    from collections import OrderedDict
    from json import dump

    hash_seed = os.environ.get('PYTHONHASHSEED', None)
    if hash_seed != '0':
        logger.warning(
            ("PYTHONHASHSEED is not 0, so the same param_dict has different "
             "hashes across sessions. Consider disabling this randomization "
             "with `PYTHONHASHSEED=0 python your_script.py`"))

    param_dict = OrderedDict(sorted(param_dict.items()))
    param_hash = str(hash(str(param_dict)))
    assert param_hash != '' # gotta be careful because of rm_if_exists

    directory = join(directory, param_hash)
    makedirs(directory, rm_if_exists=rm_if_exists)

    # Write parameters into a .json
    json_f = join(directory, 'param.json')
    with open(json_f, 'w') as h:
        dump(param_dict, h, indent=4, sort_keys=True)

    logger.info("Parameters dumped to: %s", json_f)

    return directory


def fix_terminal():
    """Fixes messed up terminal."""
    from shlex import split
    from subprocess import Popen, DEVNULL

    cmd = 'stty sane'
    child = Popen(split(cmd), stdout=DEVNULL, stderr=DEVNULL)
    _, _ = child.communicate()


def call(cmd, cwd=None, wait=True, quiet=False):
    """Executes a command in shell.

    Args:
        cmd (str): Command to be executed.
        cwd (str, optional): Directory to execute the command in. ``None``
            means current directory.
        wait (bool, optional): Whether to block until the call finishes.
        quiet (bool, optional): Whether to print out the output stream (if any)
            and error stream (if error occured).

    Returns:
        tuple:
            - **retcode** (*int*) -- Command exit code. 0 means a successful
              call. Always ``None`` if not waiting for the command to finish.
            - **stdout** (*str*) -- Standard output stream. Always ``None`` if
              not waiting.
            - **stderr** (*str*) -- Standard error stream. Always ``None`` if
              not waiting.
    """
    from subprocess import Popen, PIPE

    process = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd, shell=True)

    if not wait:
        return None, None, None

    stdout, stderr = process.communicate() # waits for completion
    stdout, stderr = stdout.decode(), stderr.decode()

    if not quiet:
        if stdout != '':
            format_print(stdout, 'O')
        if process.returncode != 0:
            if stderr != '':
                format_print(stderr, 'E')

    retcode = process.returncode
    return retcode, stdout, stderr


def _call_assert_success(cmd, **kwargs):
    retcode, stdout, _ = call(cmd, **kwargs)
    assert retcode == 0, \
        "External process call failed with exit code {code}:\n\t{cmd}".format(
            cmd=cmd, code=retcode)
    return stdout
