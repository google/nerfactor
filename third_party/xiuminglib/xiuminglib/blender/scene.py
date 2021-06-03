from os import remove
from os.path import dirname, exists

from ..imprt import preset_import

from .. import log, os as xm_os
logger = log.get_logger()


def save_blend(outpath=None, delete_overwritten=False):
    """Saves current scene to a .blend file.

    Args:
        outpath (str, optional): Path to save the scene to, e.g.,
            ``'~/foo.blend'``. ``None`` means saving to the current file.
        delete_overwritten (bool, optional): Whether to delete or keep
            as .blend1 the same-name file.

    Writes
        - A .blend file.
    """
    bpy = preset_import('bpy', assert_success=True)

    if outpath is not None:
        # "Save as" scenario: delete and then save
        xm_os.makedirs(dirname(outpath))
        if exists(outpath) and delete_overwritten:
            remove(outpath)

    try:
        # bpy.ops.file.autopack_toggle()
        bpy.ops.file.pack_all()
    except RuntimeError:
        logger.error("Failed to pack some files")

    if outpath is None:
        # "Save" scenario: save and then delete
        bpy.ops.wm.save_as_mainfile()
        outpath = bpy.context.blend_data.filepath
        bakpath = outpath + '1'
        if exists(bakpath) and delete_overwritten:
            remove(bakpath)
    else:
        bpy.ops.wm.save_as_mainfile(filepath=outpath)

    logger.info("Saved to %s", outpath)


def open_blend(inpath):
    """Opens a .blend file.

    Args:
        inpath (str): E.g., ``'~/foo.blend'``.
    """
    bpy = preset_import('bpy', assert_success=True)

    bpy.ops.wm.open_mainfile(filepath=inpath)
