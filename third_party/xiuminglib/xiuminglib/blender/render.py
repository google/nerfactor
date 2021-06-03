from os.path import dirname, join
from glob import glob
from shutil import move
from time import time

from .. import os as xm_os
from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


def set_cycles(w=None, h=None,
               n_samples=None, max_bounces=None, min_bounces=None,
               transp_bg=None,
               color_mode=None, color_depth=None):
    """Sets up Cycles as rendering engine.

    ``None`` means no change.

    Args:
        w (int, optional): Width of render in pixels.
        h (int, optional): Height of render in pixels.
        n_samples (int, optional): Number of samples.
        max_bounces (int, optional): Maximum number of light bounces.
            Setting max_bounces to 0 for direct lighting only.
        min_bounces (int, optional): Minimum number of light bounces.
        transp_bg (bool, optional): Whether world background is transparent.
        color_mode (str, optional): Color mode: ``'BW'``, ``'RGB'`` or
            ``'RGBA'``.
        color_depth (str, optional): Color depth: ``'8'`` or ``'16'``.
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    cycles.use_progressive_refine = True
    if n_samples is not None:
        cycles.samples = n_samples
    if max_bounces is not None:
        cycles.max_bounces = max_bounces
    if min_bounces is not None:
        cycles.min_bounces = min_bounces
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5

    # Ensure there's no background light emission
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    # If world background is transparent with premultiplied alpha
    if transp_bg is not None:
        render.film_transparent = transp_bg

    # # Use GPU
    # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    # bpy.context.user_preferences.system.compute_device = \
    # 'CUDA_' + str(randint(0, 3))
    # scene.cycles.device = 'GPU'

    scene.render.tile_x = 16 # 256 optimal for GPU
    scene.render.tile_y = 16 # 256 optimal for GPU
    if w is not None:
        scene.render.resolution_x = w
    if h is not None:
        scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    if color_mode is not None:
        scene.render.image_settings.color_mode = color_mode
    if color_depth is not None:
        scene.render.image_settings.color_depth = color_depth

    logger.info("Cycles set up as rendering engine")


def easyset(w=None, h=None,
            n_samples=None,
            ao=None,
            color_mode=None,
            file_format=None,
            color_depth=None,
            sampling_method=None,
            n_aa_samples=None):
    """Sets some of the scene attributes more easily.

    Args:
        w (int, optional): Width of render in pixels.
        h (int, optional): Height of render in pixels.
        n_samples (int, optional): Number of samples.
        ao (bool, optional): Ambient occlusion.
        color_mode (str, optional): Color mode of rendering: ``'BW'``,
            ``'RGB'``, or ``'RGBA'``.
        file_format (str, optional): File format of the render: ``'PNG'``,
            ``'OPEN_EXR'``, etc.
        color_depth (str, optional): Color depth of rendering: ``'8'`` or
            ``'16'`` for .png; ``'16'`` or ``'32'`` for .exr.
        sampling_method (str, optional): Method to sample light and
            materials: ``'PATH'`` or ``'BRANCHED_PATH'``.
        n_aa_samples (int, optional): Number of anti-aliasing samples (used
            with ``'BRANCHED_PATH'``).
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene

    scene.render.resolution_percentage = 100

    if w is not None:
        scene.render.resolution_x = w

    if h is not None:
        scene.render.resolution_y = h

    # Number of samples
    if n_samples is not None and scene.render.engine == 'CYCLES':
        scene.cycles.samples = n_samples

    # Ambient occlusion
    if ao is not None:
        scene.world.light_settings.use_ambient_occlusion = ao

    # Color mode of rendering
    if color_mode is not None:
        scene.render.image_settings.color_mode = color_mode

    # File format of the render
    if file_format is not None:
        scene.render.image_settings.file_format = file_format

    # Color depth of rendering
    if color_depth is not None:
        scene.render.image_settings.color_depth = color_depth

    # Method to sample light and materials
    if sampling_method is not None:
        scene.cycles.progressive = sampling_method

    # Number of anti-aliasing samples
    if n_aa_samples is not None:
        scene.cycles.aa_samples = n_aa_samples


def _render_prepare(cam, obj_names):
    bpy = preset_import('bpy', assert_success=True)

    if cam is None:
        cams = [o for o in bpy.data.objects if o.type == 'CAMERA']
        assert (len(cams) == 1), \
            "With `cam` not provided, there must be exactly one camera"
        cam = cams[0]

    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in bpy.data.objects if o.type == 'MESH']
    # Should be a list of strings by now

    for x in obj_names:
        assert isinstance(x, str), \
            ("Objects should be specified by their names (strings), not "
             "objects themselves")

    scene = bpy.context.scene

    # Set active camera
    scene.camera = cam

    # Hide objects to ignore
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = obj.name not in obj_names

    scene.use_nodes = True

    # Clear the current scene node tree to avoid unexpected renderings
    nodes = scene.node_tree.nodes
    for n in nodes:
        if n.name != "Render Layers":
            nodes.remove(n)

    outnode = nodes.new('CompositorNodeOutputFile')

    return cam.name, obj_names, scene, outnode


def _render(scene, outnode, result_socket, outpath, exr=True, alpha=True):
    bpy = preset_import('bpy', assert_success=True)

    node_tree = scene.node_tree

    # Set output file format
    if exr:
        file_format = 'OPEN_EXR'
        color_depth = '32'
        ext = '.exr'
    else:
        file_format = 'PNG'
        color_depth = '16'
        ext = '.png'
    if alpha:
        color_mode = 'RGBA'
    else:
        color_mode = 'RGB'

    outnode.base_path = '/tmp/%s/' % time()
    # NOTE: The trailing slash is important

    # Connect result socket(s) to the output node
    if isinstance(result_socket, dict):
        assert exr, ".exr must be used for multi-layer results"
        file_format += '_MULTILAYER'
        assert 'composite' in result_socket.keys(), (
            "Composite pass is always rendered anyways. Plus, we need this "
            "dummy connection for the multi-layer OpenEXR file to be saved "
            "to disk (strangely)")
        node_tree.links.new(result_socket['composite'], outnode.inputs['Image'])
        # Add input slots and connect
        for k, v in result_socket.items():
            outnode.layer_slots.new(k)
            node_tree.links.new(v, outnode.inputs[k])
        render_f = join(outnode.base_path, '????.exr')

    else:
        node_tree.links.new(result_socket, outnode.inputs['Image'])
        render_f = join(outnode.base_path, 'Image????' + ext)

    outnode.format.file_format = file_format
    outnode.format.color_depth = color_depth
    outnode.format.color_mode = color_mode

    scene.render.filepath = '/tmp/%s' % time() # composite (to discard)

    # Render
    bpy.ops.render.render(write_still=True)

    # Depending on the scene state, the render filename may be anything
    # matching the pattern
    fs = glob(render_f)
    assert len(fs) == 1, (
        "There should be only one file matching:\n\t{p}\n"
        "but found {n}").format(p=render_f, n=len(fs))
    render_f = fs[0]

    # Move from temporary directory to the desired location
    if not outpath.endswith(ext):
        outpath += ext
    move(render_f, outpath)
    return outpath


def render(outpath, cam=None, obj_names=None, alpha=True, text=None):
    """Renders current scene with cameras in scene.

    Args:
        outpath (str): Path to save the render to. Should end with either
            .exr or .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, use the only camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. If ``None``, all objects are of interest and will
            appear in the render.
        alpha (bool, optional): Whether to render the alpha channel.
        text (dict, optional): What text to be overlaid on image and how,
            following the format::

                {
                    'contents': "Hello World!",
                    'bottom_left_corner': (50, 50),
                    'font_scale': 1,
                    'bgr': (255, 0, 0),
                    'thickness': 2,
                }

    Writes
        - A 32-bit .exr or 16-bit .png image.
    """
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    result_socket = scene.node_tree.nodes['Render Layers'].outputs['Image']

    # Render
    exr = outpath.endswith('.exr')
    outpath = _render(
        scene, outnode, result_socket, outpath, exr=exr, alpha=alpha)

    # Optionally overlay text
    if text is not None:
        cv2 = preset_import('cv2')
        im = cv2.imread(outpath, cv2.IMREAD_UNCHANGED)
        cv2.putText(
            im, text['contents'], text['bottom_left_corner'],
            cv2.FONT_HERSHEY_SIMPLEX, text['font_scale'], text['bgr'],
            text['thickness'])
        cv2.imwrite(outpath, im)

    logger.info("%s rendered through '%s'", obj_names, cam_name)
    logger.warning(
        "Node trees and renderability of these objects have changed")


def _disable_cycles_mat_nodes_for_bi():
    """Disables Cycles material's nodes for Blender Internal.

    Cycles use_nodes being True leads to 0 alpha in Blender Internal.
    """
    bpy = preset_import('bpy', assert_success=True)

    if bpy.context.scene.render.engine == 'BLENDER_RENDER':
        for o in bpy.data.objects:
            mat = o.active_material
            if mat is not None and mat.use_nodes:
                mat.use_nodes = False


def render_depth(outprefix, cam=None, obj_names=None, ray_depth=False):
    r"""Renders raw depth map in .exr of the specified object(s) from the
    specified camera.

    The EXR data contain an aliased :math:`z` map and an anti-aliased alpha
    map.

    Args:
        outprefix (str): Where to save the .exr maps to, e.g., ``'~/depth'``.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be the just one camera in the
            scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        ray_depth (bool, optional): Whether to render ray or plane depth.

    Writes
        - A 32-bit .exr depth map w/o anti-aliasing, located at
          ``outprefix + '_z.exr'``.
        - A 32-bit .exr alpha map w/ anti-aliasing, located at
          ``outprefix + '_a.exr'``.

    Todo:
        Ray depth.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    if ray_depth:
        raise NotImplementedError("Ray depth")

    # Use Blender Render for anti-aliased results -- faster than Cycles,
    # which needs >1 samples to figure out object boundary
    scene.render.engine = 'BLENDER_RENDER'
    scene.render.alpha_mode = 'TRANSPARENT'
    _disable_cycles_mat_nodes_for_bi()

    node_tree = scene.node_tree
    nodes = node_tree.nodes

    # Render z pass, without anti-aliasing to avoid values interpolated
    # between real depth values (e.g., 1.5) and large background depth values
    # (e.g., 1e10)
    scene.render.use_antialiasing = False
    scene.use_nodes = True
    try:
        result_socket = nodes['Render Layers'].outputs['Z']
    except KeyError:
        result_socket = nodes['Render Layers'].outputs['Depth']
    outpath_z = _render(scene, outnode, result_socket, outprefix + '_z')

    # Render alpha pass, with anti-aliasing to get a soft mask for blending
    scene.render.use_antialiasing = True
    result_socket = nodes['Render Layers'].outputs['Alpha']
    outpath_a = _render(scene, outnode, result_socket, outprefix + '_a')

    logger.info("Depth map of %s rendered through '%s' to",
                obj_names, cam_name)
    logger.info("\t1. z w/o anti-aliasing: %s", outpath_z)
    logger.info("\t2. alpha w/ anti-aliasing: %s", outpath_a)
    logger.warning("The scene node tree has changed")


def render_alpha(outpath, cam=None, obj_names=None, samples=1000):
    r"""Renders binary or soft mask of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    scene.render.engine = 'CYCLES'
    film_transparent_old = scene.render.film_transparent
    scene.render.film_transparent = True
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    result_socket = nodes['Render Layers'].outputs['Alpha']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=False, alpha=False)

    # Restore
    scene.cycles.samples = samples_old
    scene.render.film_transparent = film_transparent_old

    logger.info(
        "Foreground alpha of %s rendered through '%s'", obj_names, cam_name)
    logger.warning(
        "Node trees and renderability of these objects have changed")


def render_normal(outpath, cam=None, obj_names=None,
                  outpath_refball=None, world_coords=False):
    r"""Renders raw normal map in .exr of the specified object(s) from the
    specified camera.

    RGB at each pixel is the (almost unit) normal vector at that location.

    Args:
        outpath (str): The .exr path (so data are raw values, not integer
            values) we save the normal map to.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be only one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        outpath_refball (str, optional): The .exr path to save the reference
            ball's normals to. ``None`` means not rendering the reference
            ball.
        world_coords (bool, optional): Whether to render normals in the world
            or *negated* camera space.

    Warning:
        **TL;DR**

        If you want camera-space normal maps, you need to negate
        the normal map after loading it from the .exr file this function
        writes. Otherwise, those normals live in a space that has all three
        axes flipped w.r.t. the camera's local space.

        **The Details**

        If you want world-space normals, then easy; I'll just use
        Cycles, and the normals are automatically in the world space. If
        camera-space normals are what you want, I'll use Blender Internal
        (BI), but there's some complication taken care of under the hood.

        BI renders normals "in the camera space." I verified this by keeping
        my scene intact, but having my camera rotating a bit; indeed, the
        normal vectors of a cube went from "round values", such as
        :math:`(0, 0, 1)`, to "non-round values", such as
        :math:`(0.02, 0.03, 0.99)`.

        But what precisely is this "camera space" (denoted by :math:`S`)? Is
        it really just the camera's local space? How do we go from :math:`S`
        to the world coordinate system, and possibly to another space
        therefrom? Here's my exploration.

        I put a camera at the scene center, and had it pointing, head-on, to
        one face of a default cube (so the camera saw just that face -- no
        other faces). I rendered the normals with BI: the raw RGB value is
        :math:`(0, 0, −1)`. OK, so the normal vector pointing out of the
        screen to my eyes is :math:`S`'s :math:`−z`. Hence, :math:`S`'s
        :math:`+z` points into the screen.

        Then I rotated the cube by just a little, so my camera got to see a
        little bit of the side faces it couldn't see before. The normal vector
        pointing to the left is :math:`(1, 0, 0)`; so the :math:`+x` of
        :math:`S` points to the left, tangent to the screen. Similarly, I
        found out the :math:`+y` points downwards, also tangent to the screen.

        But wait, the three axes don't even form a right-handed system; they
        form a left-handed one! This is so strange. Oh, if we negate all the
        axes, then we get a right-handed system. Would this negated system be
        the camera's local space (as in an object's local coordinate system)?

        It is! After eyeballing the camera's local space axes, I found they
        are exactly the flipped axes of :math:`S`. Therefore, when
        camera-space normals are asked for, I first render them out using BI,
        and then have to flip the signs to give the camera-space normals,
        which you can then transform to other spaces correctly with
        transformation matrices.

    Writes
        - A 32-bit .exr normal map of the object(s) of interest.
        - Another 32-bit .exr normal map of the reference ball, if asked for.
    """
    from .object import add_sphere, remove_objects
    from .camera import get_2d_bounding_box
    bpy = preset_import('bpy', assert_success=True)

    objs = bpy.data.objects

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    cam = objs[cam_name]

    # # Make normals consistent
    # for obj_name in obj_names:
    #     scene.objects.active = objs[obj_name]
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all()
    #     bpy.ops.mesh.normals_make_consistent()
    #     bpy.ops.object.mode_set(mode='OBJECT')

    # Set up scene node tree
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    render_layer = bpy.context.view_layer # currently active one
    render_layer.use_pass_normal = True
    set_alpha_node = nodes.new('CompositorNodeSetAlpha')
    node_tree.links.new(
        nodes['Render Layers'].outputs['Alpha'], set_alpha_node.inputs['Alpha'])
    node_tree.links.new(
        nodes['Render Layers'].outputs['Normal'],
        set_alpha_node.inputs['Image'])
    result_socket = set_alpha_node.outputs['Image']

    # Select rendering engine based on whether camera or world space
    if world_coords:
        scene.render.engine = 'CYCLES'
        film_transparent_old = scene.render.film_transparent
        scene.render.film_transparent = True
        samples_old = scene.cycles.samples
        scene.cycles.samples = 16 # for anti-aliased edges
    else: # camera space
        scene.render.engine = 'BLENDER_RENDER'
        scene.render.alpha_mode = 'TRANSPARENT'
        _disable_cycles_mat_nodes_for_bi()

    def add_refball():
        from .camera import get_camera_matrix
        from ..geometry.proj import from_homo
        Vector = preset_import('Vector')
        # Place the ball at any depth so long as its center projects to
        # image center
        z_c = 10 # any reasonable depth in the camera space
        cam_mat, cam_int, _ = get_camera_matrix(cam, keep_disparity=True)
        x, y = cam_int[0][2], cam_int[1][2]
        center_xy1d = Vector([x, y, 1, 1 / z_c]) # with disparity
        xyzw = cam_mat.inverted() @ center_xy1d # from pixel to world
        xyz = from_homo(xyzw)
        sphere = add_sphere(location=xyz)
        # Scale the ball so that it, when projected, fits into the frame
        # and occupies reasonably large portion of the image
        bbox = get_2d_bounding_box(sphere, cam)
        s = 0.8 / max((bbox[1, 0] - bbox[0, 0]) / (2 * x),
                      (bbox[3, 1] - bbox[0, 1]) / (2 * y))
        sphere.scale = (s, s, s)
        # Achieve smooth normals with low polycount
        for f in sphere.data.polygons:
            f.use_smooth = True
        return sphere

    def render_refball(refball, out):
        mesh_hide_render = {}
        # Hide everything but the ball
        for o in [x for x in objs if x.type == 'MESH']:
            mesh_hide_render[o.name] = o.hide_render # save old state
            o.hide_render = o.name != refball.name
        out = _render(scene, outnode, result_socket, out)
        # Restore hide_render
        for k, v in mesh_hide_render.items():
            objs[k].hide_render = v
        remove_objects(refball.name)
        return out

    if outpath_refball is not None:
        # Add reference normal ball
        refball = add_refball()
        outpath_refball = render_refball(refball, outpath_refball)

    # Render
    outpath = _render(scene, outnode, result_socket, outpath)

    # Restore
    if world_coords:
        scene.render.film_transparent = film_transparent_old
        scene.cycles.samples = samples_old

    logger.info("Normal map of %s rendered through %s to %s",
                obj_names, cam_name, outpath)
    if outpath_refball is not None:
        logger.info("Renference ball rendered through the same camera to %s",
                    outpath_refball)
    logger.warning("The scene node tree has changed")


def render_lighting_passes(
        outpath, cam=None, obj_names=None, n_samples=None, select=None):
    """Renders select Cycles' lighting passes of the specified object(s) from
    the specified camera.

    Data are in a single multi-layer .exr file. See the code below for what
    channels are rendered.

    Args:
        outpath (str): Where to save the lighting passes to. Should end with
            .exr.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be only one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        n_samples (int, optional): Number of samples per pixel. Useful when
            you want a value different than other renderings; ``None`` means
            using the current value.
        select (list(str), optional): Render only this list of passes.
            ``None`` means rendering all passes: ``diffuse_direct``,
            ``diffuse_indirect``, ``diffuse_color``, ``glossy_direct``,
            ``glossy_indirect``, and ``glossy_color``.

    Writes
        - A 32-bit .exr multi-layer image containing the lighting passes.
    """
    bpy = preset_import('bpy', assert_success=True)

    all_passes = { # a set useful for intrinsic image decomposition
        'diffuse_direct': 'DiffDir', 'diffuse_indirect': 'DiffInd',
        'diffuse_color': 'DiffCol', 'glossy_direct': 'GlossDir',
        'glossy_indirect': 'GlossDir', 'glossy_color': 'GlossCol'}
    if select is None:
        select_passes = all_passes
    elif isinstance(select, str):
        select_passes = {select: all_passes[select]}
    else:
        select_passes = {k: v for k, v in all_passes.items() if k in select}

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    scene.render.engine = 'CYCLES'
    n_samples_old = scene.cycles.samples
    if n_samples is not None:
        scene.cycles.samples = n_samples
    film_transparent_old = scene.render.film_transparent
    scene.render.film_transparent = True

    # Enable all passes of interest
    render_layer = bpy.context.view_layer # currently active one
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    result_sockets = {}
    for p, p_key in select_passes.items():
        setattr(render_layer, 'use_pass_' + p, True)
        # Set alpha
        set_alpha_node = nodes.new('CompositorNodeSetAlpha')
        node_tree.links.new(
            nodes['Render Layers'].outputs['Alpha'],
            set_alpha_node.inputs['Alpha'])
        node_tree.links.new(
            nodes['Render Layers'].outputs[p_key],
            set_alpha_node.inputs['Image'])
        result_sockets[p] = set_alpha_node.outputs['Image']

    # Render
    if len(result_sockets) == 1:
        # NOTE: To avoid multi-layer .exr so that one can just read the render
        # with OpenCV (multi-layer would require OpenEXR)
        result_sockets = list(result_sockets.values())[0]
    else:
        # Multi-layer .exr saving strangely requires this composite pass
        result_sockets['composite'] = nodes['Render Layers'].outputs['Image']
    outpath = _render(scene, outnode, result_sockets, outpath)

    # Restore
    scene.cycles.samples = n_samples_old
    scene.render.film_transparent = film_transparent_old

    logger.info(
        "Select lighting passes of %s rendered through '%s' to %s",
        obj_names, cam_name, outpath)
    logger.warning("The scene node tree has changed")
