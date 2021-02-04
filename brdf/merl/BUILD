py_library(
    name = "merl",
    srcs = ["merl.py"],
    deps = [
        "//experimental/users/xiuming/sim/third_party/nielsen2015on",
        "//experimental/users/xiuming/xiuminglib",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        # "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "render",
    srcs = ["render.py"],
    python_version = "PY3",
    deps = [
        ":merl",
        "//experimental/users/xiuming/sim/brdf:renderer",
        "//experimental/users/xiuming/xiuminglib",
        "//file/colossus/cns",
        "//pyglib:gfile",
        "//third_party/py/numpy",
        "//third_party/py/tqdm",
    ],
)

py_binary(
    name = "normalize",
    srcs = ["normalize.py"],
    python_version = "PY3",
    deps = [
        ":merl",
        "//experimental/users/xiuming/xiuminglib",
        "//file/colossus/cns",
        "//pyglib:gfile",
        "//third_party/py/numpy",
        "//third_party/py/tqdm",
    ],
)
