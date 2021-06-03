import re
from os.path import basename, dirname
import numpy as np

from ..imprt import preset_import

from .. import log, os as xm_os
logger = log.get_logger()


def get_object(otype, any_ok=False):
    """Gets the handle of the only (or any) object of the given type.

    Args:
        otype (str): Object type: ``'MESH'``, ``'CAMERA'``, ``'LAMP'`` or any
            string ``a_bpy_obj.type`` may return.
        any_ok (bool, optional): Whether it's ok to grab any object when there
            exist multiple ones matching the given type. If ``False``, there
            must be exactly one object of the given type.

    Returns:
        bpy_types.Object.
    """
    bpy = preset_import('bpy', assert_success=True)

    objs = [x for x in bpy.data.objects if x.type == otype]
    n_objs = len(objs)
    if n_objs == 0:
        raise RuntimeError("There's no object matching the given type")
    if n_objs == 1:
        return objs[0]
    # More than one objects
    if any_ok:
        return objs[0]
    raise RuntimeError((
        "When `any_ok` is `False`, there must be exactly "
        "one object matching the given type"))


def remove_objects(name_pattern, regex=False):
    """Removes object(s) from current scene.

    Args:
        name_pattern (str): Name or name pattern of object(s) to remove.
        regex (bool, optional): Whether to interpret ``name_pattern`` as a
            regex.
    """
    bpy = preset_import('bpy', assert_success=True)

    objs = bpy.data.objects
    removed = []

    if regex:
        assert (name_pattern != '*'), \
            "Want to match everything? Correct regex for that is '.*'"

        name_pattern = re.compile(name_pattern)

        for obj in objs:
            if name_pattern.match(obj.name):
                obj.select_set(True)
                removed.append(obj.name)
            else:
                obj.select_set(False)

    else:
        for obj in objs:
            if obj.name == name_pattern:
                obj.select_set(True)
                removed.append(obj.name)
            else:
                obj.select_set(False)

    # Delete
    bpy.ops.object.delete()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Removed from scene: %s", removed)


def import_object(
        model_path, axis_forward='-Z', axis_up='Y',
        rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        trans_vec=(0, 0, 0), scale=1, merge=False, name=None):
    """Imports external object to current scene, the low-level way.

    Args:
        model_path (str): Path to object to add.
        axis_forward (str, optional): Which direction is forward.
        axis_up (str, optional): Which direction is upward.
        rot_mat (array_like, optional): 3-by-3 rotation matrix *preceding*
            translation.
        trans_vec (array_like, optional): 3D translation vector *following*
            rotation.
        scale (float, optional): Scale of the object.
        merge (bool, optional): Whether to merge objects into one.
        name (str, optional): Object name after import.

    Returns:
        bpy_types.Object or list(bpy_types.Object): Imported object(s).
    """
    bpy = preset_import('bpy', assert_success=True)
    Matrix = preset_import('Matrix', assert_success=True)

    # Deselect all
    for o in bpy.data.objects:
        o.select_set(False)

    # Import
    if model_path.endswith('.obj'):
        bpy.ops.import_scene.obj(
            filepath=model_path, axis_forward=axis_forward, axis_up=axis_up)
    elif model_path.endswith('.ply'):
        bpy.ops.import_mesh.ply(filepath=model_path)
        logger.warning("axis_forward and axis_up ignored for .ply")
    else:
        raise NotImplementedError(".%s" % model_path.split('.')[-1])

    # Merge, if asked to
    if merge and len(bpy.context.selected_objects) > 1:
        objs_to_merge = bpy.context.selected_objects
        context = bpy.context.copy()
        context['active_object'] = objs_to_merge[0]
        context['selected_objects'] = objs_to_merge
        context['selected_editable_bases'] = \
            [bpy.context.scene.object_bases[o.name] for o in objs_to_merge]
        bpy.ops.object.join(context)
        objs_to_merge[0].name = 'merged' # change object name
        # objs_to_merge[0].data.name = 'merged' # change mesh name

    obj_list = []
    for i, obj in enumerate(bpy.context.selected_objects):

        # Rename
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # Compute world matrix
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # don't scale here
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4

        # Scale
        obj.scale = (scale, scale, scale)

        obj_list.append(obj)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Imported: %s", model_path)

    if len(obj_list) == 1:
        return obj_list[0]
    return obj_list


def export_object(obj_names, model_path, axis_forward=None, axis_up=None):
    """Exports Blender object(s) to a file.

    Args:
        obj_names (str or list(str)): Object name(s) to export. Must be a
            single string if output format is .ply.
        model_path (str): Output .obj or .ply path.
        axis_forward (str, optional): Which direction is forward. For .obj,
            the default is ``'-Z'``, and ``'Y'`` for .ply.
        axis_up (str, optional): Which direction is upward. For .obj, the
            default is ``'Y'``, and ``'Z'`` for .ply.

    Writes
        - Exported model file, possibly accompanied by a material file.
    """
    bpy = preset_import('bpy', assert_success=True)

    out_dir = dirname(model_path)
    xm_os.makedirs(out_dir)

    if isinstance(obj_names, str):
        obj_names = [obj_names]

    exported = []
    for o in [x for x in bpy.data.objects if x.type == 'MESH']:
        o.select_set(o.name in obj_names)
        if o.select:
            exported.append(o.name)

    if model_path.endswith('.ply'):
        assert len(obj_names) == 1, \
            ".ply holds a single object; use .obj for multiple objects"

        if axis_forward is None:
            axis_forward = 'Y'
        if axis_up is None:
            axis_up = 'Z'

        bpy.ops.export_mesh.ply(
            filepath=model_path, axis_forward=axis_forward, axis_up=axis_up)

    elif model_path.endswith('.obj'):
        if axis_forward is None:
            axis_forward = '-Z'
        if axis_up is None:
            axis_up = 'Y'

        bpy.ops.export_scene.obj(
            filepath=model_path, use_selection=True,
            axis_forward=axis_forward, axis_up=axis_up)

    else:
        raise NotImplementedError(".%s" % model_path.split('.')[-1])

    logger.info("%s Exported to %s", exported, model_path)


def add_cylinder_between(pt1, pt2, r=1e-3, name=None):
    """Adds a cylinder specified by two end points and radius.

    Super useful for visualizing rays in ray tracing while debugging.

    Args:
        pt1 (array_like): World coordinates of point 1.
        pt2 (array_like): World coordinates of point 2.
        r (float, optional): Cylinder radius.
        name (str, optional): Cylinder name.

    Returns:
        bpy_types.Object: Cylinder added.
    """
    bpy = preset_import('bpy', assert_success=True)

    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    d = pt2 - pt1

    # Add cylinder at the correct location
    dist = np.linalg.norm(d)
    loc = (pt1[0] + d[0] / 2, pt1[1] + d[1] / 2, pt1[2] + d[2] / 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=dist, location=loc)

    cylinder_obj = bpy.context.object

    if name is not None:
        cylinder_obj.name = name

    # Further rotate it accordingly
    phi = np.arctan2(d[1], d[0])
    theta = np.arccos(d[2] / dist)
    cylinder_obj.rotation_euler[1] = theta
    cylinder_obj.rotation_euler[2] = phi

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    return cylinder_obj


def add_rectangular_plane(
        center_loc=(0, 0, 0), point_to=(0, 0, 1), size=(2, 2), name=None):
    """Adds a rectangular plane specified by its center location, dimensions,
    and where its +z points to.

    Args:
        center_loc (array_like, optional): Plane center location in world
            coordinates.
        point_to (array_like, optional): Point in world coordinates to which
            plane's +z points.
        size (array_like, optional): Sizes in x and y directions (0 in z).
        name (str, optional): Plane name.

    Returns:
        bpy_types.Object: Plane added.
    """
    bpy = preset_import('bpy', assert_success=True)
    Vector = preset_import('Vector', assert_success=True)

    center_loc = np.array(center_loc)
    point_to = np.array(point_to)
    size = np.append(np.array(size), 0)

    bpy.ops.mesh.primitive_plane_add(location=center_loc)

    plane_obj = bpy.context.object

    if name is not None:
        plane_obj.name = name

    plane_obj.dimensions = size

    # Point it to target
    direction = Vector(point_to) - plane_obj.location
    # Find quaternion that rotates plane's 'Z' so that it aligns with
    # `direction`. This rotation is not unique because the rotated plane can
    # still rotate about direction vector. Specifying 'Y' gives the rotation
    # quaternion with plane's 'Y' pointing up
    rot_quat = direction.to_track_quat('Z', 'Y')
    plane_obj.rotation_euler = rot_quat.to_euler()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    return plane_obj


def create_mesh(verts, faces, name='new-mesh'):
    """Creates a mesh from vertices and faces.

    Args:
        verts (array_like): Local coordinates of the vertices, of shape
            N-by-3.
        faces (list(tuple)): Faces specified by ordered vertex indices.
        name (str, optional): Mesh name.

    Returns:
        bpy_types.Mesh: Mesh data created.
    """
    bpy = preset_import('bpy', assert_success=True)

    verts = np.array(verts)

    # Create mesh
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    logger.info("Mesh '%s' created", name)

    return mesh_data


def create_object_from_mesh(mesh_data, obj_name='new-obj',
                            location=(0, 0, 0), rotation_euler=(0, 0, 0),
                            scale=(1, 1, 1)):
    """Creates object from mesh data.

    Args:
        mesh_data (bpy_types.Mesh): Mesh data.
        obj_name (str, optional): Object name.
        location (tuple, optional): Object location in world coordinates.
        rotation_euler (tuple, optional): Object rotation in radians.
        scale (tuple, optional): Object scale.

    Returns:
        bpy_types.Object: Object created.
    """
    bpy = preset_import('bpy', assert_success=True)

    # Create
    obj = bpy.data.objects.new(obj_name, mesh_data)

    # Link to current scene
    scene = bpy.context.scene
    scene.objects.link(obj)
    obj.select_set(True)
    scene.objects.active = obj # make the selection effective

    # Set attributes
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.scale = scale

    logger.info("Object '%s' created from mesh data and selected", obj_name)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    return obj


def _clear_nodetree_for_active_material(obj):
    """Internal helper function clears the node tree of active material.

    So that desired node tree can be cleanly set up. If no active material, one
    will be created.
    """
    bpy = preset_import('bpy', assert_success=True)

    # Create material if none
    if obj.active_material is None:
        mat = bpy.data.materials.new(name='new-mat-for-%s' % obj.name)
        if obj.data.materials:
            # Assign to first material slot
            obj.data.materials[0] = mat
        else:
            # No slots
            obj.data.materials.append(mat)

    active_mat = obj.active_material
    active_mat.use_nodes = True
    node_tree = active_mat.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    return node_tree


def color_vertices(obj, vert_ind, colors):
    r"""Colors each vertex of interest with the given color.

    Colors are defined for vertex loops, in fact. This function uses the same
    color for all loops of a vertex. Useful for making a 3D heatmap.

    Args:
        obj (bpy_types.Object): Object.
        vert_ind (int or list(int)): Index/indices of vertex/vertices to
            color.
        colors (tuple or list(tuple)): RGB value(s) to paint on
            vertex/vertices. Values :math:`\in [0, 1]`. If one tuple,
            this color will be applied to all vertices. If list of tuples,
            must be of the same length as ``vert_ind``.
    """
    bpy = preset_import('bpy', assert_success=True)

    # Validate inputs
    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]
    else:
        vert_ind = list(vert_ind)
    if isinstance(colors, tuple):
        colors = [colors] * len(vert_ind)
    assert (len(colors) == len(vert_ind)), \
        ("`colors` and `vert_ind` must be of the same length, "
         "or `colors` is a single tuple")
    for i, c in enumerate(colors):
        c = tuple(c)
        if len(c) == 3:
            colors[i] = c + (1,)
        elif len(c) == 4: # In case some Blender version needs 4-tuples
            colors[i] = c
        else:
            raise ValueError("Wrong color length: %d" % len(c))
    if any(x > 1 for c in colors for x in c):
        logger.warning("Did you forget to normalize color values to [0, 1]?")

    scene = bpy.context.scene
    scene.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data

    if mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.active
    else:
        vcol_layer = mesh.vertex_colors.new()

    # A vertex and one of its edges combined are called a loop, which has a
    # color. So if a vertex has four outgoing edges, it has four colors for
    # the four loops
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop_vert_idx = mesh.loops[loop_idx].vertex_index
            try:
                color_idx = vert_ind.index(loop_vert_idx)
            except ValueError:
                color_idx = None
            if color_idx is not None:
                try:
                    vcol_layer.data[loop_idx].color = colors[color_idx]
                except ValueError:
                    # This Blender version requires 3-tuples
                    vcol_layer.data[loop_idx].color = colors[color_idx][:3]

    # Set up nodes for vertex colors
    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes
    attr_node = nodes.new('ShaderNodeAttribute')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    nodes['Attribute'].attribute_name = vcol_layer.name
    node_tree.links.new(attr_node.outputs[0], diffuse_node.inputs[0])
    node_tree.links.new(diffuse_node.outputs[0], output_node.inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Vertex color(s) added to '%s'", obj.name)
    logger.warning("..., so node tree of '%s' has changed", obj.name)


def _assert_cycles(scene):
    engine = scene.render.engine
    if engine != 'CYCLES':
        raise NotImplementedError(engine)


def _make_texture_node(obj, texture_str):
    bpy = preset_import('bpy', assert_success=True)

    mat = obj.active_material
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    texture_node = nodes.new('ShaderNodeTexImage')
    if texture_str == 'bundled':
        texture = mat.active_texture
        assert texture is not None, "No bundled texture found"
        img = texture.image
    else:
        # Path given -- external texture map
        bpy.data.images.load(texture_str, check_existing=True)
        img = bpy.data.images[basename(texture_str)]
        # Careless texture mapping
        texture_node.projection = 'FLAT'
        texcoord_node = nodes.new('ShaderNodeTexCoord')
        if obj.data.uv_layers.active is None:
            texcoord = texcoord_node.outputs['Generated']
        else:
            texcoord = texcoord_node.outputs['UV']
        node_tree.links.new(texcoord, texture_node.inputs['Vector'])
    texture_node.image = img
    return texture_node


def setup_simple_nodetree(obj, texture, shader_type, roughness=0):
    r"""Sets up a simple (diffuse and/or glossy) node tree.

    Texture can be an bundled texture map, a path to an external texture map,
    or simply a pure color. If a path to an external image, and UV
    coordinates are given (e.g., in the geometry .obj file), then they will be
    used. If they are not given, texture mapping will be done carelessly,
    with automatically generated UV coordinates. See private function
    :func:`_make_texture_node` for how this is done.

    Args:
        obj (bpy_types.Object): Object, optionally bundled with texture map.
        texture (str or tuple): If string, must be ``'bundled'`` or path to
            the texture image. If tuple, must be of 4 floats
            :math:`\in [0, 1]` as RGBA values.
        shader_type (str): Either ``'diffuse'`` or ``'glossy'``.
        roughness (float, optional): If diffuse, the roughness in Oren-Nayar,
            0 gives Lambertian. If glossy, 0 means perfectly reflective.
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    if shader_type == 'glossy':
        shader_node = nodes.new('ShaderNodeBsdfGlossy')
    elif shader_type == 'diffuse':
        shader_node = nodes.new('ShaderNodeBsdfDiffuse')
    else:
        raise ValueError(shader_type)

    if isinstance(texture, str):
        texture_node = _make_texture_node(obj, texture)
        node_tree.links.new(
            texture_node.outputs['Color'], shader_node.inputs['Color'])
    elif isinstance(texture, tuple):
        shader_node.inputs['Color'].default_value = texture
    else:
        raise TypeError(texture)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(
        shader_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Roughness
    shader_node.inputs['Roughness'].default_value = roughness

    logger.info(
        "%s node tree set up for '%s'", shader_type.capitalize(), obj.name)


def setup_emission_nodetree(obj, texture=(1, 1, 1, 1), strength=1, hide=False):
    r"""Sets up an emission node tree for the object.

    Args:
        obj (bpy_types.Object): Object (maybe bundled with texture map).
        texture (str or tuple, optional): If string, must be ``'bundled'`` or
            path to the texture image. If tuple, must be of 4 floats
            :math:`\in [0, 1]` as RGBA values.
        strength (float, optional): Emission strength.
        hide (bool, optional): Useful for hiding the emissive object (but
            keeping the light of course).
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    # Emission node
    nodes.new('ShaderNodeEmission')
    if isinstance(texture, str):
        texture_node = _make_texture_node(obj, texture)
        node_tree.links.new(
            texture_node.outputs['Color'], nodes['Emission'].inputs['Color'])
    elif isinstance(texture, tuple):
        nodes['Emission'].inputs['Color'].default_value = texture
    else:
        raise TypeError(texture)
    nodes['Emission'].inputs['Strength'].default_value = strength

    # Output node
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(
        nodes['Emission'].outputs['Emission'],
        nodes['Material Output'].inputs['Surface'])

    # hide_render hides the object and the light, but this keeps the light
    obj.cycles_visibility.camera = not hide

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Emission node tree set up for '%s'", obj.name)


def setup_holdout_nodetree(obj):
    """Sets up a holdout node tree for the object.

    Args:
        obj (bpy_types.Object): Object bundled with texture map.
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    nodes.new('ShaderNodeHoldout')
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(
        nodes['Holdout'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Holdout node tree set up for '%s'", obj.name)


def setup_retroreflective_nodetree(
        obj, texture, roughness=0, glossy_weight=0.1):
    r"""Sets up a retroreflective texture node tree.

    Bundled texture can be an external texture map (carelessly mapped) or a
    pure color. Mathematically, the BRDF model is a mixture of a diffuse BRDF
    and a glossy BRDF using incoming light directions as normals.

    Args:
        obj (bpy_types.Object): Object, optionally bundled with texture map.
        texture (str or tuple): If string, must be ``'bundled'`` or path to
            the texture image. If tuple, must be of 4 floats :math:`\in [0, 1]`
            as RGBA values.
        roughness (float, optional): Roughness for both the glossy and diffuse
            shaders.
        glossy_weight (float, optional): Mixture weight for the glossy shader.
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    # Set color for diffuse and glossy nodes
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    glossy_node = nodes.new('ShaderNodeBsdfGlossy')
    if isinstance(texture, str):
        texture_node = _make_texture_node(obj, texture)
        node_tree.links.new(
            texture_node.outputs['Color'], diffuse_node.inputs['Color'])
        node_tree.links.new(
            texture_node.outputs['Color'], glossy_node.inputs['Color'])
    elif isinstance(texture, tuple):
        diffuse_node.inputs['Color'].default_value = texture
        glossy_node.inputs['Color'].default_value = texture
    else:
        raise TypeError(texture)

    geometry_node = nodes.new('ShaderNodeNewGeometry')
    mix_node = nodes.new('ShaderNodeMixShader')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(
        geometry_node.outputs['Incoming'], glossy_node.inputs['Normal'])
    node_tree.links.new(diffuse_node.outputs['BSDF'], mix_node.inputs[1])
    node_tree.links.new(glossy_node.outputs['BSDF'], mix_node.inputs[2])
    node_tree.links.new(
        mix_node.outputs['Shader'], output_node.inputs['Surface'])

    # Roughness
    diffuse_node.inputs['Roughness'].default_value = roughness
    glossy_node.inputs['Roughness'].default_value = roughness

    mix_node.inputs['Fac'].default_value = glossy_weight

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Retroreflective node tree set up for '%s'", obj.name)


def get_bmesh(obj):
    """Gets Blender mesh data from object.

    Args:
        obj (bpy_types.Object): Object.

    Returns:
        BMesh: Blender mesh data.
    """
    bpy = preset_import('bpy', assert_success=True)
    bmesh = preset_import('bmesh', assert_success=True)

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    return bm


def subdivide_mesh(obj, n_subdiv=2):
    """Subdivides mesh of object.

    Args:
        obj (bpy_types.Object): Object whose mesh is to be subdivided.
        n_subdiv (int, optional): Number of subdivision levels.
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene

    # All objects need to be in 'OBJECT' mode to apply modifiers -- maybe a
    # Blender bug?
    for o in bpy.data.objects:
        scene.objects.active = o
        bpy.ops.object.mode_set(mode='OBJECT')
        o.select_set(False)
    obj.select_set(True)
    scene.objects.active = obj

    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers['Subdivision'].subdivision_type = 'CATMULL_CLARK'
    obj.modifiers['Subdivision'].levels = n_subdiv
    obj.modifiers['Subdivision'].render_levels = n_subdiv

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier='Subdivision', apply_as='DATA')

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Subdivided mesh of '%s'", obj.name)


def select_mesh_elements_by_vertices(obj, vert_ind, select_type):
    """Selects vertices or their associated edges/faces in edit mode.

    Args:
        obj (bpy_types.Object): Object.
        vert_ind (int or list(int)): Vertex index/indices.
        select_type (str): Type of mesh elements to select: ``'vertex'``,
            ``'edge'`` or ``'face'``.
    """
    bpy = preset_import('bpy', assert_success=True)
    bmesh = preset_import('bmesh', assert_success=True)

    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]

    # Edit mode
    scene = bpy.context.scene
    scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Deselect all
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')

    bm = bmesh.from_edit_mesh(obj.data)
    bvs = bm.verts

    bvs.ensure_lookup_table()
    for i in vert_ind:
        bv = bvs[i]

        if select_type == 'vertex':
            bv.select_set(True)

        # Select all edges with this vertex at an end
        elif select_type == 'edge':
            for be in bv.link_edges:
                be.select_set(True)

        # Select all faces with this vertex
        elif select_type == 'face':
            for bf in bv.link_faces:
                bf.select_set(True)

        else:
            raise ValueError("Wrong selection type")

    # Update viewport
    scene.objects.active = scene.objects.active

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    logger.info("Selected %s elements of '%s'", select_type, obj.name)


def add_sphere(
        location=(0, 0, 0), scale=1, n_subdiv=2, shade_smooth=False, name=None):
    """Adds a sphere.

    Args:
        location (array_like, optional): Location of the sphere center.
        scale (float, optional): Scale of the sphere.
        n_subdiv (int, optional): Control of how round the sphere is.
        shade_smooth (bool, optional): Whether to use smooth shading.
        name (str, optional): Name of the added sphere.

    Returns:
        bpy_types.Object: Sphere created.
    """
    bpy = preset_import('bpy', assert_success=True)

    bpy.ops.mesh.primitive_ico_sphere_add()
    sphere = bpy.context.active_object

    if name is not None:
        sphere.name = name

    sphere.location = location
    sphere.scale = (scale, scale, scale)

    # Subdivide for smoother sphere
    bpy.ops.object.modifier_add(type='SUBSURF')
    sphere.modifiers['Subdivision'].subdivision_type = 'CATMULL_CLARK'
    sphere.modifiers['Subdivision'].levels = n_subdiv
    sphere.modifiers['Subdivision'].render_levels = n_subdiv
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.modifier_apply(modifier='Subdivision', apply_as='DATA')

    # Fake smoothness
    if shade_smooth:
        for f in sphere.data.polygons:
            f.use_smooth = True

    return sphere


def smart_uv_unwrap(obj, area_weight=0.0):
    """UV unwrapping using Blender's smart projection.

    A vertex may map to multiple UV locations, but each loop maps to exactly
    one UV location. If a face uses M vertices, then it has M loops, so a vertex
    may belong to multiple loops, each of which has one UV location.

    Note:
        If a vertex belongs to no face, it doesn't get a UV coordinate,
        so don't assume you can get a UV for any given vertex index.

    Args:
        obj (bpy_types.Object): Object to UV unwrap.
        area_weight (float, optional): Area weight.

    Returns:
        dict(numpy.ndarray): Dictionary with its keys being the face indices,
        and values being 2D arrays with four columns containing the
        corresponding face's loop indices, vertex indices, :math:`u`, and
        :math:`v`.

        UV coordinate convention:

        .. code-block:: none

            (0, 1)
                ^ v
                |
                |
                |
                |
                +-----------> (1, 0)
            (0, 0)        u
    """
    bpy = preset_import('bpy', assert_success=True)

    assert obj.type == 'MESH'

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(user_area_weight=area_weight)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Since # faces is usually very large, using faces as dictionary
    # keys usually leads to speedups (compared with having them as
    # an array's column and then slicing the array)
    fi_li_vi_u_v = {}
    for f in obj.data.polygons:
        li_vi_u_v = []
        for vi, li in zip(f.vertices, f.loop_indices):
            uv = obj.data.uv_layers.active.data[li].uv
            li_vi_u_v.append([li, vi, uv.x, uv.y])
        fi_li_vi_u_v[f.index] = np.array(li_vi_u_v)

    return fi_li_vi_u_v


def raycast(obj_bvhtree, ray_from_objspc, ray_to_objspc):
    """Casts a ray to an object.

    Args:
        obj_bvhtree (mathutils.bvhtree.BVHTree): Constructed BVH tree of the
            object.
        ray_from_objspc (mathutils.Vector): Ray origin, in object's local
            coordinates.
        ray_to_objspc (mathutils.Vector): Ray goes through this point, also
            specified in the object's local coordinates. Note that the ray
            doesn't stop at this point, and this is just for computing the
            ray direction.

    Returns:
        tuple:
            - **hit_loc** (*mathutils.Vector*) -- Hit location on the object,
              in the object's local coordinates. ``None`` means no
              intersection.
            - **hit_normal** (*mathutils.Vector*) -- Normal of the hit
              location, also in the object's local coordinates.
            - **hit_fi** (*int*) -- Index of the face where the hit happens.
            - **ray_dist** (*float*) -- Distance that the ray has traveled
              before hitting the object. If ``ray_to_objspc`` is a point on
              the object surface, then this return value is useful for
              checking for self occlusion.
    """
    ray_dir = (ray_to_objspc - ray_from_objspc).normalized()
    hit_loc, hit_normal, hit_fi, ray_dist = \
        obj_bvhtree.ray_cast(ray_from_objspc, ray_dir)
    if hit_loc is None:
        assert hit_normal is None and hit_fi is None and ray_dist is None
    return hit_loc, hit_normal, hit_fi, ray_dist
