# For blaze on Google's infrastructure

py_library(
    name = "xiuminglib",
    srcs = glob([
        "*.py",
    ]),
    srcs_version = "PY3",
    deps = [
        "//experimental/users/xiuming/xiuminglib/data",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/blender",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/geometry",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/io",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/vis",
        "//learning/deepmind/video/python:video_api",
        "//pyglib:gfile",
        "//file/colossus/cns",
        "//third_party/py/cvx2",
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/axes_grid1",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//third_party/py/numpy",
        "//third_party/py/PIL:pil",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
        "//third_party/py/tensorflow",
        # bpy not in third_party, so has to be wrapped into an MPM package
        # NOTE: Listing OpenEXR and Imath as deps here segfaults bpy
    ],
)

py_library(
    name = "exr",
    srcs = glob([
        "*.py",
    ]),
    srcs_version = "PY3",
    deps = [
        "//experimental/users/xiuming/xiuminglib/data",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/blender",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/geometry",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/io",
        "//experimental/users/xiuming/xiuminglib/xiuminglib/vis",
        "//learning/deepmind/video/python:video_api",
        "//pyglib:gfile",
        "//file/colossus/cns",
        "//third_party/py/cvx2",
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/axes_grid1",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//third_party/py/numpy",
        "//third_party/py/PIL:pil",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
        "//third_party/py/Imath",
        "//third_party/py/OpenEXR",
        "//third_party/py/tensorflow",
        # bpy not in third_party, so has to be wrapped into an MPM package
    ],
)
