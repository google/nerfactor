from ..imprt import preset_import


def cursor_to(loc):
    """Moves the cursor to the given 3D location.

    Useful for inspecting where a 3D point is in the scene, to do which you
    first use this function, save the scene, and open the scene in GUI.

    Args:
        loc (array_like): 3D coordinates, of length 3.
    """
    bpy = preset_import('bpy', assert_success=True)

    bpy.context.scene.cursor.location = loc
