py_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
        "//experimental/users/xiuming/sim/sim/util",
        "//third_party/google_research/google_research/robust_loss:adaptive",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "trainvali",
    srcs = ["trainvali.py"],
    data = glob(["config/*.ini"]),
    python_version = "PY3",
    deps = [
        "//experimental/users/xiuming/sim/sim/datasets",
        "//experimental/users/xiuming/sim/sim/models",
        "//experimental/users/xiuming/sim/sim/util",
        "//learning/deepmind/xmanager2/client/google",
        "//perftools/accelerators/xprof/xprofilez/cpu:python_tracer",
        "//perftools/gputools/profiler:gpuprof_lib",
        "//perftools/gputools/profiler:xprofilez_with_server",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/absl:app",
        "//third_party/py/tensorflow",
        "//third_party/py/tqdm",
    ],
)

py_binary(
    name = "grab_latest_ckpt",
    srcs = ["grab_latest_ckpt.py"],
    python_version = "PY3",
    deps = [
        "//experimental/users/xiuming/sim/sim/util",
        "//pyglib:gfile",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "sim_test",
    srcs = ["sim_test.py"],
    data = glob(["config/*.ini"]),
    python_version = "PY3",
    deps = [
        "//experimental/users/xiuming/sim/sim/datasets",
        "//experimental/users/xiuming/sim/sim/models",
        "//experimental/users/xiuming/sim/sim/util",
        "//experimental/users/xiuming/xiuminglib",
        "//pipeline/flume/py:runner",
        "//pyglib:gfile",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/apache_beam",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tqdm",
    ],
)
