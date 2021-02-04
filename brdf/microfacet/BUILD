py_library(
    name = "microfacet",
    srcs = ["microfacet.py"],
    deps = [
        "//experimental/users/xiuming/sim/sim/util",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "render",
    srcs = ["render.py"],
    python_version = "PY3",
    deps = [
        ":microfacet",
        "//experimental/users/xiuming/sim/brdf:renderer",
        "//experimental/users/xiuming/xiuminglib",
        "//file/colossus/cns",
        "//pyglib:gfile",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "across_rough",
    srcs = ["across_rough.py"],
    python_version = "PY3",
    deps = [
        ":microfacet",
        "//experimental/users/xiuming/sim/brdf:renderer",
        "//experimental/users/xiuming/xiuminglib",
        "//file/colossus/cns",
        "//pyglib:gfile",
        "//third_party/py/tensorflow",
        "//third_party/py/tqdm",
    ],
)
