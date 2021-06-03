from os import remove, rename
from os.path import dirname, basename
from time import time
import numpy as np

from .object import get_bmesh, raycast
from ..imprt import preset_import
from ..io import exr
from ..geometry.proj import from_homo

from ..log import get_logger
logger = get_logger()


def add_camera(xyz=(0, 0, 0),
               rot_vec_rad=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=100):
    """Adds a camera to  the current scene.

    Args:
        xyz (tuple, optional): Location. Defaults to ``(0, 0, 0)``.
        rot_vec_rad (tuple, optional): Rotations in radians around x, y and z.
            Defaults to ``(0, 0, 0)``.
        name (str, optional): Camera object name.
        proj_model (str, optional): Camera projection model. Must be
            ``'PERSP'``, ``'ORTHO'``, or ``'PANO'``. Defaults to ``'PERSP'``.
        f (float, optional): Focal length in mm. Defaults to 35.
        sensor_fit (str, optional): Sensor fit. Must be ``'HORIZONTAL'`` or
            ``'VERTICAL'``. See also :func:`get_camera_matrix`. Defaults to
            ``'HORIZONTAL'``.
        sensor_width (float, optional): Sensor width in mm. Defaults to 32.
        sensor_height (float, optional): Sensor height in mm. Defaults to 18.
        clip_start (float, optional): Near clipping distance. Defaults to 0.1.
        clip_end (float, optional): Far clipping distance. Defaults to 100.

    Returns:
        bpy_types.Object: Camera added.
    """
    bpy = preset_import('bpy', assert_success=True)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end

    logger.info("Camera '%s' added", cam.name)

    return cam


def easyset(cam,
            xyz=None,
            rot_vec_rad=None,
            name=None,
            proj_model=None,
            f=None,
            sensor_fit=None,
            sensor_width=None,
            sensor_height=None):
    """Sets camera parameters more easily.

    See :func:`add_camera` for arguments. ``None`` will result in no change.
    """
    if name is not None:
        cam.name = name

    if xyz is not None:
        cam.location = xyz

    if rot_vec_rad is not None:
        cam.rotation_euler = rot_vec_rad

    if proj_model is not None:
        cam.data.type = proj_model

    if f is not None:
        cam.data.lens = f

    if sensor_fit is not None:
        cam.data.sensor_fit = sensor_fit

    if sensor_width is not None:
        cam.data.sensor_width = sensor_width

    if sensor_height is not None:
        cam.data.sensor_height = sensor_height


def point_camera_to(cam, xyz_target, up=(0, 0, 1)):
    """Points camera to target.

    Args:
        cam (bpy_types.Object): Camera object.
        xyz_target (array_like): Target point in world coordinates.
        up (array_like, optional): World vector that, when projected,
            points up in the image plane.
    """
    Vector = preset_import('Vector', assert_success=True)
    Quaternion = preset_import('Quaternion', assert_success=True)

    failed_ensuring_up_msg = \
        "Camera '%s' pointed to %s, but with no guarantee on up vector" \
        % (cam.name, tuple(xyz_target))

    up = Vector(up)
    xyz_target = Vector(xyz_target)

    direction = xyz_target - cam.location

    # Rotate camera with quaternion so that `track` aligns with `direction`, and
    # world +z, when projected, aligns with camera +y (i.e., points up in image
    # plane)
    track = '-Z'
    rot_quat = direction.to_track_quat(track, 'Y')
    cam.rotation_euler = (0, 0, 0)
    cam.rotation_euler.rotate(rot_quat)

    # Further rotate camera so that world `up`, when projected, points up on
    # image plane. We know right now world +z, when projected, points up, so
    # we just need to rotate the camera around the lookat direction by an angle
    cam_mat, _, _ = get_camera_matrix(cam)
    up_proj = cam_mat * up.to_4d()
    orig_proj = cam_mat * Vector((0, 0, 0)).to_4d()
    try:
        up_proj = Vector((up_proj[0] / up_proj[2], up_proj[1] / up_proj[2])) - \
            Vector((orig_proj[0] / orig_proj[2], orig_proj[1] / orig_proj[2]))
    except ZeroDivisionError:
        logger.error(
            ("w in homogeneous coordinates is 0; "
             "camera coincides with the point to project? "
             "So can't rotate camera to ensure up vector"))
        logger.info(failed_ensuring_up_msg)
        return cam
    if up_proj.length == 0:
        logger.error(
            ("Up vector projected to zero length; "
             "optical axis coincides with the up vector? "
             "So can't rotate camera to ensure up vector"))
        logger.info(
            "Camera '%s' pointed to %s, but with no guarantee on up vector",
            cam.name, tuple(xyz_target))
        return cam
    # +------->
    # |
    # |
    # v
    up_proj[1] = -up_proj[1]
    # ^
    # |
    # |
    # +------->
    a = Vector((0, 1)).angle_signed(up_proj) # clockwise is positive
    cam.rotation_euler.rotate(Quaternion(direction, a))

    logger.info("Camera '%s' pointed to %s with world %s pointing up",
                cam.name, tuple(xyz_target), tuple(up))

    return cam


def intrinsics_compatible_with_scene(cam, eps=1e-6):
    r"""Checks if camera intrinsic parameters are comptible with the current
    scene.

    Intrinsic parameters include sensor size and pixel aspect ratio, and scene
    parameters refer to render resolutions and their scale. The entire sensor is
    assumed active.

    Args:
        cam (bpy_types.Object): Camera object
        eps (float, optional): :math:`\epsilon` for numerical comparison.
            Considered equal if :math:`\frac{|a - b|}{b} < \epsilon`.

    Returns:
        bool: Check result.
    """
    bpy = preset_import('bpy', assert_success=True)

    # Camera
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Do these parameters make sense together?
    mm_per_pix_horizontal = sensor_width_mm / (w * scale)
    mm_per_pix_vertical = sensor_height_mm / (h * scale)

    if abs(mm_per_pix_horizontal / mm_per_pix_vertical - pixel_aspect_ratio) \
            / pixel_aspect_ratio < eps:
        logger.info("OK")
        return True

    logger.error((
        "Render resolutions (w_pix = %d; h_pix = %d), active sensor size "
        "(w_mm = %f; h_mm = %f), and pixel aspect ratio (r = %f) don't make "
        "sense together. This could cause unexpected behaviors later. "
        "Consider running correct_sensor_height()"
    ), w, h, sensor_width_mm, sensor_height_mm, pixel_aspect_ratio)
    return False


def correct_sensor_height(cam):
    r"""To make render resolutions, sensor size, and pixel aspect ratio
    comptible.

    If render resolutions are :math:`(w_\text{pix}, h_\text{pix})`, sensor sizes
    are :math:`(w_\text{mm}, h_\text{mm})`, and pixel aspect ratio is :math:`r`,
    then
    :math:`h_\text{mm}\leftarrow\frac{h_\text{pix}}{w_\text{pix}r}w_\text{mm}`.

    Args:
        cam (bpy_types.Object): Camera.
    """
    bpy = preset_import('bpy', assert_success=True)

    # Camera
    sensor_width_mm = cam.data.sensor_width

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    pixel_aspect_ratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Change sensor height
    sensor_height_mm = sensor_width_mm * h / w / pixel_aspect_ratio
    cam.data.sensor_height = sensor_height_mm

    logger.info("Sensor height changed to %f", sensor_height_mm)


def get_camera_matrix(cam, keep_disparity=False):
    r"""Gets camera matrix, intrinsics, and extrinsics from a camera.

    You can ask for a 4-by-4 projection that projects :math:`(x, y, z, 1)` to
    :math:`(x, y, 1, d)`, where :math:`d` is the disparity, reciprocal of
    depth.

    ``cam_mat.dot(pts)`` gives you projections in the following convention:

    .. code-block:: none

        +------------>
        |       proj[:, 0]
        |
        |
        v proj[:, 1]

    Args:
        cam (bpy_types.Object): Camera.
        keep_disparity (bool, optional): Whether or not the matrices keep
            disparity.

    Returns:
        tuple:
            - **cam_mat** (*mathutils.Matrix*) -- Camera matrix, product of
              intrinsics and extrinsics. 4-by-4 if ``keep_disparity``; else,
              3-by-4.
            - **int_mat** (*mathutils.Matrix*) -- Camera intrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-3.
            - **ext_mat** (*mathutils.Matrix*) -- Camera extrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-4.
    """
    bpy = preset_import('bpy', assert_success=True)
    Matrix = preset_import('Matrix', assert_success=True)

    bpy.context.view_layer.update()

    # Check if camera intrinsic parameters comptible with render settings
    if not intrinsics_compatible_with_scene(cam):
        raise ValueError(
            ("Render settings and camera intrinsic parameters mismatch. "
             "Such computed matrices will not make sense. Make them "
             "consistent first. See error message from "
             "intrinsics_compatible_with_scene() above for advice"))

    # Intrinsics

    f_mm = cam.data.lens
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if cam.data.sensor_fit == 'VERTICAL':
        # h times pixel height must fit into sensor_height_mm
        # w / pixel_aspect_ratio times pixel width will then fit into
        # sensor_width_mm
        s_y = h * scale / sensor_height_mm
        s_x = w * scale / pixel_aspect_ratio / sensor_width_mm
    else: # 'HORIZONTAL' or 'AUTO'
        # w times pixel width must fit into sensor_width_mm
        # h * pixel_aspect_ratio times pixel height will then fit into
        # sensor_height_mm
        s_x = w * scale / sensor_width_mm
        s_y = h * scale * pixel_aspect_ratio / sensor_height_mm

    skew = 0 # only use rectangular pixels

    if keep_disparity:
        # 4-by-4
        int_mat = Matrix((
            (s_x * f_mm, skew, w * scale / 2, 0),
            (0, s_y * f_mm, h * scale / 2, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1)))
    else:
        # 3-by-3
        int_mat = Matrix((
            (s_x * f_mm, skew, w * scale / 2),
            (0, s_y * f_mm, h * scale / 2),
            (0, 0, 1)))

    # Extrinsics

    # Three coordinate systems involved:
    #   1. World coordinates "world"
    #   2. Blender camera coordinates "cam":
    #        - x is horizontal
    #        - y is up
    #        - right-handed: negative z is look-at direction
    #   3. Desired computer vision camera coordinates "cv":
    #        - x is horizontal
    #        - y is down (to align to the actual pixel coordinates)
    #        - right-handed: positive z is look-at direction

    rotmat_cam2cv = Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)))

    # matrix_world defines local-to-world transformation, i.e.,
    # where is local (x, y, z) in world coordinate system?
    t, rot_euler = cam.matrix_world.decompose()[0:2]

    # World to Blender camera
    rotmat_world2cam = rot_euler.to_matrix().transposed() # same as inverse
    t_world2cam = rotmat_world2cam @ -t

    # World to computer vision camera
    rotmat_world2cv = rotmat_cam2cv @ rotmat_world2cam
    t_world2cv = rotmat_cam2cv @ t_world2cam

    if keep_disparity:
        # 4-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],),
            (0, 0, 0, 1)))
    else:
        # 3-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],)))

    # Camera matrix
    cam_mat = int_mat @ ext_mat

    logger.info("Done computing camera matrix for '%s'", cam.name)
    logger.warning("... using w = %d; h = %d", w * scale, h * scale)

    return cam_mat, int_mat, ext_mat


def get_camera_zbuffer(cam, save_to=None, hide=None):
    """Gets :math:`z`-buffer of the camera.

    Values are :math:`z` components in camera-centered coordinate system,
    where

    - :math:`x` is horizontal;
    - :math:`y` is down (to align with the actual pixel coordinates);
    - right-handed: positive :math:`z` is look-at direction and means
      "in front of camera."

    Origin is the camera center, not image plane (one focal length away
    from origin).

    Args:
        cam (bpy_types.Object): Camera.
        save_to (str, optional): Path to which the .exr :math:`z`-buffer will
            be saved. ``None`` means don't save.
        hide (str or list(str)): Names of objects to be hidden while rendering
            this camera's :math:`z`-buffer.

    Returns:
        numpy.ndarray: Camera :math:`z`-buffer.
    """
    bpy = preset_import('bpy', assert_success=True)
    cv2 = preset_import('cv2', assert_success=True)

    # Validate and standardize error-prone inputs
    if hide is not None:
        if not isinstance(hide, list):
            # A single object
            hide = [hide]
        for element in hide:
            assert isinstance(element, str), \
                ("`hide` should contain object names (i.e., strings), "
                 "not objects themselves")

    if save_to is None:
        outpath = '/tmp/%s_zbuffer' % time()
    elif save_to.endswith('.exr'):
        outpath = save_to[:-4]

    # Duplicate scene to avoid touching the original scene
    bpy.ops.scene.new(type='LINK_OBJECTS')

    scene = bpy.context.scene
    scene.camera = cam
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    # Set up nodes for z pass
    rlayers_node = nodes.new('CompositorNodeRLayers')
    output_node = nodes.new('CompositorNodeOutputFile')
    node_tree.links.new(rlayers_node.outputs[2], output_node.inputs[0])
    output_node.format.file_format = 'OPEN_EXR'
    output_node.format.color_mode = 'RGB'
    output_node.format.color_depth = '32' # full float
    output_node.base_path = dirname(outpath)
    output_node.file_slots[0].path = basename(outpath)

    # Hide objects from z-buffer, if necessary
    if hide is not None:
        orig_hide_render = {} # for later restoration
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                orig_hide_render[obj.name] = obj.hide_render
                obj.hide_render = obj.name in hide

    # Render
    scene.cycles.samples = 1
    scene.render.filepath = '/tmp/%s_rgb.png' % time() # to avoid overwritting
    bpy.ops.render.render(write_still=True)

    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Delete this new scene
    bpy.ops.scene.delete()

    # Restore objects' original render hide states, if necessary
    if hide is not None:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.hide_render = orig_hide_render[obj.name]

    # Load z-buffer as array
    exr_path = outpath + '%04d' % scene.frame_current + '.exr'
    im = exr.read(exr_path)
    assert (
        np.array_equal(im[:, :, 0], im[:, :, 1]) and np.array_equal(
            im[:, :, 0], im[:, :, 2])), (
                "BGR channels of the z-buffer should be all the same, "
                "but they are not")
    zbuffer = im[:, :, 0]

    # Delete or move the .exr as user wants
    if save_to is None:
        # User doesn't want it -- delete
        remove(exr_path)
    else:
        # User wants it -- rename
        rename(exr_path, outpath + '.exr')

    logger.info("Got z-buffer of camera '%s'", cam.name)
    logger.warning("... using w = %d; h = %d", w * scale, h * scale)

    return zbuffer


def backproject_to_3d(xys, cam, obj_names=None, world_coords=False):
    """Backprojects 2D coordinates to 3D.

    Since a 2D point could have been projected from any point on a 3D line,
    this function will return the 3D point at which this line (ray)
    intersects with an object for the first time.

    Args:
        xys (array_like): XY coordinates of length 2 or shape N-by-2,
            in the following convention:

            .. code-block:: none

                (0, 0)
                +------------> (w, 0)
                |           x
                |
                |
                |
                v y (0, h)

        cam (bpy_types.Object): Camera.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means considering all objects.
        world_coords (bool, optional): Whether to return world or the object's
            local coordinates.

    Returns:
        tuple:
            - **ray_tos** (*mathutils.Vector or list(mathutils.Vector)*) --
              Location(s) at which each ray points in the world coordinates,
              regardless of ``world_coords``. This and the (shared) ray origin
              (``cam.location``) determine the rays.
            - **xyzs** (*mathutils.Vector or list(mathutils.Vector)*) --
              Intersection coordinates specified in either the world or the
              object's local coordinates, depending on ``world_coords``.
              ``None`` means no intersection.
            - **intersect_objnames** (*str or list(str)*) -- Name(s) of
              object(s) responsible for intersections. ``None`` means no
              intersection.
            - **intersect_facei** (*int or list(int)*) -- Index/indices of the
              face(s), where the intersection happens.
            - **intersect_normals** (*mathutils.Vector or
              list(mathutils.Vector)*) -- Normal vector(s) at the
              intersection(s) specified in the same space as ``xyzs``.
    """
    from tqdm import tqdm
    bpy = preset_import('bpy', assert_success=True)
    Vector = preset_import('Vector', assert_success=True)
    BVHTree = preset_import('BVHTree', assert_success=True)

    # Standardize inputs
    xys = np.array(xys).reshape(-1, 2)
    objs = bpy.data.objects
    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in objs if o.type == 'MESH']
    z_c = 1 # any depth in the camera space, so long as not infinity

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get 4-by-4 invertible camera matrix
    cam_mat, _, _ = get_camera_matrix(cam, keep_disparity=True)
    cam_mat_inv = cam_mat.inverted() # pixel space to world

    # Precompute BVH trees and world-to-object transformations
    trees, world2objs = {}, {}
    for obj_name in obj_names:
        world2objs[obj_name] = objs[obj_name].matrix_world.inverted()
        obj = objs[obj_name]
        bm = get_bmesh(obj)
        trees[obj_name] = BVHTree.FromBMesh(bm)

    ray_tos = [None] * xys.shape[0]
    xyzs = [None] * xys.shape[0]
    intersect_objnames = [None] * xys.shape[0]
    intersect_facei = [None] * xys.shape[0]
    intersect_normals = [None] * xys.shape[0]

    ray_from_world = cam.location

    # TODO: vectorize for performance
    for i, xy in enumerate(tqdm(xys, desc="Computing ray internsections")):

        # Compute any point on the line passing camera center and
        # projecting to (x, y)
        xy1d = np.append(xy, [1, 1 / z_c]) # with disparity
        xyzw = cam_mat_inv @ Vector(xy1d) # world

        # Ray start and direction in world coordinates
        ray_to_world = from_homo(xyzw)
        ray_tos[i] = ray_to_world

        first_intersect = None
        first_intersect_objname = None
        first_intersect_facei = None
        first_intersect_normal = None
        dist_min = np.inf

        # Test intersections with each object of interest
        for obj_name, tree in trees.items():
            obj2world = objs[obj_name].matrix_world
            world2obj = world2objs[obj_name]

            # Ray start and direction in local coordinates
            ray_from = world2obj @ ray_from_world
            ray_to = world2obj @ ray_to_world

            # Ray tracing
            loc, normal, facei, _ = raycast(tree, ray_from, ray_to)
            # Not using the returned ray distance as that's local
            dist = None if loc is None else (
                obj2world @ loc - ray_from_world).length

            # See if this intersection is closer to camera center than
            # previous intersections with other objects
            if (dist is not None) and (dist < dist_min):
                first_intersect = obj2world @ loc if world_coords else loc
                first_intersect_objname = obj_name
                first_intersect_facei = facei
                # NOTE: local-to-world transformation of normals should not
                # apply translations, since normals are "anchor-less," unlike
                # locations
                first_intersect_normal = \
                    obj2world.to_3x3() @ normal if world_coords else normal
                first_intersect_normal.normalize()
                # Re-normalize in case transforming to world coordinates has
                # ruined the unit length
                dist_min = dist

        xyzs[i] = first_intersect
        intersect_objnames[i] = first_intersect_objname
        intersect_facei[i] = first_intersect_facei
        intersect_normals[i] = first_intersect_normal

    assert None not in ray_tos, (
        "No matter whether a ray is a hit or not, we must have a "
        "\"look-at\" for it")

    logger.info("Backprojection done with camera '%s'", cam.name)
    logger.warning("... using w = %d; h = %d", w * scale, h * scale)

    ret = (
        ray_tos, xyzs, intersect_objnames, intersect_facei, intersect_normals)
    if xys.shape[0] == 1:
        return tuple(x[0] for x in ret)
    return ret


def get_visible_vertices(cam, obj, ignore_occlusion=False, hide=None,
                         method='raycast', perc_eps=1e-6):
    r"""Gets vertices that are visible (projected within frame *and*
    unoccluded) from camera.

    Args:
        cam (bpy_types.Object): Camera.
        obj (bpy_types.Object): Object of interest.
        ignore_occlusion (bool, optional): Whether to ignore all occlusion
            (including self-occlusion). Useful for finding out which vertices
            fall inside the camera view.
        hide (str or list(str), optional): Names of objects to be hidden
            while rendering this camera's :math:`z`-buffer. No effect if
            ``ignore_occlusion``.
        method (str, optional): Visibility test method: ``'raycast'`` or
            ``'zbuffer'``. Ray casting is more robust than comparing the
            vertex's depth against :math:`z`-buffer (inaccurate when the
            render resolution is low, or when object's own depth variation is
            small compared with its overall depth). The advantage of the
            :math:`z`-buffer, though, is its runtime independent of number
            of vertices.
        perc_eps (float, optional): Threshold for percentage difference
            between test value :math:`x` and true value :math:`y`. :math:`x`
            is considered equal to :math:`y` when :math:`\frac{|x - y|}{y}`
            is smaller. No effect if ``ignore_occlusion``.

    Returns:
        list: Indices of vertices that are visible.
    """
    bpy = preset_import('bpy', assert_success=True)
    bmesh = preset_import('bmesh', assert_success=True)
    BVHTree = preset_import('BVHTree', assert_success=True)

    legal = ('zbuffer', 'raycast')
    assert method in legal, \
        "Legal methods: %s, but found '%s'" % (legal, method)

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get camera matrix
    cam_mat, _, ext_mat = get_camera_matrix(cam)

    # Get z-buffer
    if not ignore_occlusion:
        if method == 'zbuffer':
            zbuffer = get_camera_zbuffer(cam, hide=hide)
        else:
            zbuffer = None

    # Get mesh data from object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if zbuffer is None:
        tree = BVHTree.FromBMesh(bm)
        world2obj = obj.matrix_world.inverted()
        ray_from = world2obj * cam.location # object's local coordinates

    def are_close(x, y):
        return (x - y) / y < perc_eps

    visible_vert_ind = []
    # For each of its vertices
    # TODO: vectorize for speed
    for bv in bm.verts:

        # Check if its projection falls inside frame
        v_world = obj.matrix_world * bv.co # local to world
        xy = np.array(cam_mat * v_world) # project to 2D
        xy = xy[:-1] / xy[-1]
        if xy[0] >= 0 and xy[0] < w * scale and \
                xy[1] >= 0 and xy[1] < h * scale:
            # Falls into the camera view

            if ignore_occlusion:
                # Considered visible already
                visible = True
            else:
                # Check occlusion
                if zbuffer is None:
                    # ... by raycasting
                    ray_to = bv.co # local coordinates
                    ray_dist_no_occlu = (ray_to - ray_from).length
                    _, _, _, ray_dist = raycast(tree, ray_from, ray_to)
                    visible = ray_dist is not None and \
                        are_close(ray_dist_no_occlu, ray_dist)
                else:
                    # ... by comparing against z-buffer
                    v_cv = ext_mat * v_world # world to camera to CV
                    z = v_cv[-1]
                    z_min = zbuffer[int(xy[1]), int(xy[0])]
                    visible = are_close(z, z_min)

            if visible:
                visible_vert_ind.append(bv.index)

    logger.info("Visibility test done with camera '%s'", cam.name)
    logger.warning("... using w = %d; h = %d", w * scale, h * scale)

    return visible_vert_ind


def get_2d_bounding_box(obj, cam):
    """Gets a 2D bounding box of the object in the camera frame.

    This is different from projecting the 3D bounding box to 2D.

    Args:
        obj (bpy_types.Object): Object of interest.
        cam (bpy_types.Object): Camera.

    Returns:
        numpy.ndarray: 2D coordinates of the bounding box corners.
        Of shape 4-by-2. Corners are ordered counterclockwise, following:

        .. code-block:: none

            (0, 0)
            +------------> (w, 0)
            |           x
            |
            |
            |
            v y (0, h)
    """
    bpy = preset_import('bpy', assert_success=True)

    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100.
    w = scene.render.resolution_x * scale
    h = scene.render.resolution_y * scale

    # Get camera matrix
    cam_mat, _, _ = get_camera_matrix(cam)

    # Project all vertices to 2D
    pts = np.vstack([v.co.to_4d() for v in obj.data.vertices]).T # 4-by-N
    world_mat = np.array(obj.matrix_world) # 4-by-4
    cam_mat = np.array(cam_mat) # 3-by-4
    xyw = cam_mat.dot(world_mat.dot(pts)) # 3-by-N
    pts_2d = np.divide(xyw[:2, :], np.tile(xyw[2, :], (2, 1))) # 2-by-N

    # Compute bounding box
    x_min, y_min = np.min(pts_2d, axis=1)
    x_max, y_max = np.max(pts_2d, axis=1)
    corners = np.vstack((
        np.array([x_min, y_min]),
        np.array([x_max, y_min]),
        np.array([x_max, y_max]),
        np.array([x_min, y_max])))

    logger.info("Got 2D bounding box of '%s' in camera '%s'",
                obj.name, cam.name)
    logger.warning("... using w = %d; h = %d", w, h)

    if x_min < 0 or x_max >= w or y_min < 0 or y_max >= h:
        logger.warning("Part of the bounding box falls outside the frame")

    return corners
