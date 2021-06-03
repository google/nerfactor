from importlib import import_module

from .log import get_logger
logger = get_logger()

# For < Python 3.6
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError


def preset_import(name, assert_success=False):
    """A unified importer for both regular and ``google3`` modules, according
    to specified presets/profiles (e.g., ignoring ``ModuleNotFoundError``).
    """
    if name in ('cv2', 'opencv'):
        try:
            # BUILD dep:
            # "//third_party/py/cvx2",
            from cvx2 import latest as mod
            # Or
            # BUILD dep:
            # "//third_party/OpenCVX:cvx2",
            # from google3.third_party.OpenCVX import cvx2 as cv2
        except ModuleNotFoundError:
            mod = import_module_404ok('cv2')

    elif name in ('tf', 'tensorflow'):
        mod = import_module_404ok('tensorflow')

    elif name == 'gfile':
        # BUILD deps:
        # "//pyglib:gfile",
        # "//file/colossus/cns",
        mod = import_module_404ok('google3.pyglib.gfile')

    elif name == 'video_api':
        # BUILD deps:
        # "//learning/deepmind/video/python:video_api",
        mod = import_module_404ok(
            'google3.learning.deepmind.video.python.video_api')

    elif name in ('bpy', 'bmesh', 'OpenEXR', 'Imath'):
        # BUILD deps:
        # "//third_party/py/Imath",
        # "//third_party/py/OpenEXR",
        mod = import_module_404ok(name)

    elif name in ('Vector', 'Matrix', 'Quaternion'):
        mod = import_module_404ok('mathutils')
        mod = _get_module_class(mod, name)

    elif name == 'BVHTree':
        mod = import_module_404ok('mathutils.bvhtree')
        mod = _get_module_class(mod, name)

    else:
        raise NotImplementedError(name)

    if assert_success:
        assert mod is not None, "Failed in importing '%s'" % name

    return mod


def import_module_404ok(*args, **kwargs):
    """Returns ``None`` (instead of failing) in the case of
    ``ModuleNotFoundError``.
    """
    try:
        mod = import_module(*args, **kwargs)
    except (ModuleNotFoundError, ImportError) as e:
        mod = None
        logger.debug("Ignored: %s", str(e))
    return mod


def _get_module_class(mod, clsname):
    if mod is None:
        return None
    return getattr(mod, clsname)
