# SIM: Data Generation

This folder provides the code for rendering your own data. You do not need this
if you use our rendered data (available in "Downloads -> Rendered Data" of
[the project page](http://nlt.csail.mit.edu)).

`render.py` is the core script that renders a given camera-light configuration.
`gen_render_params_expects.py` generates parameters that define different
camera-light configurations (arguments to `render.py`), for you to distribute
`render.py` over multiple machines or a render farm to render all
configurations in parallel. `get_neighbors.py` is the script that generated the
JSON files indicating the nearest neighbor for each camera/light in the metadata
.zip.

Scenes are specified in Blender and rendered with Cycles, Blender's built-in
physically-based rendering engine.


## Setup

You should use the Python bundled inside Blender, rather than that of your
system or environment.

Because of the API changes in Blender 2.8x, please run our code with
Blender 2.8x. More specifically, we used Blender 2.83.4.

1. Clone this repository:
    ```
    cd "$ROOT"
    git clone https://github.com/xiumingzhang/sim.git
    ```

1. "Install" Blender-Python (the binaries are pre-built, so just download
   and unzip):
    ```
    mkdir "$ROOT"/software
    cd "$ROOT"/software

    # Download
    wget https://download.blender.org/release/Blender2.83/blender-2.83.4-linux64.tar.xz

    # Unzip the pre-built binaries
    tar -xvf blender-2.83.4-linux64.tar.xz
    ```

1. Install the dependencies to this *Blender-bundled* Python:
    ```
    cd blender-2.83.4-linux64/2.83/python/bin

    # Install pip for THIS Blender-bundled Python
    curl https://bootstrap.pypa.io/get-pip.py | ./python3.7m
    # If errors, make sure you deactivate your Conda environment

    # Use THIS pip to install other dependencies
    ./pip install absl-py tqdm ipython numpy Pillow opencv-python
    ```

1. Make sure this Python can locate `xiuminglib`:
    ```
    export PYTHONPATH="$ROOT"/sim/third_party/xiuminglib/:$PYTHONPATH
    ```


## Rendering

There are header instructions in all the main scripts, but the general workflow
is as follows.

1. Make sure that you can render a single camera-light configuration:
    ```
    "$ROOT"/software/blender-2.83.4-linux64/blender \
        --background \
        --python "$ROOT"/sim/data_gen/render.py \
        -- \
        --scene="$ROOT"/data/scenes/dragon_specular.blend \
        --cam_json="$ROOT"/data/trainvali_cams/P28R.json \
        --light_json="$ROOT"/data/trainvali_lights/l330.json \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='512' \
        --spp='256' \
        --outdir="$ROOT"/data/scenes/dragon_specular_imh512_uvs512_spp256/trainvali_000020852_P28R_l330
    ```

1. Generate all camera-light configurations (rendering jobs) you want to render:
    ```
    python "$ROOT"/sim/data_gen/gen_render_params_expects.py \
        --mode='trainvali+test' \
        --scene="$ROOT"/data/scenes/dragon_specular.blend \
        --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
        --test_cams="$ROOT"'/data/test_cams/*.json' \
        --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
        --test_lights="$ROOT"'/data/test_lights/*.json' \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='512' \
        --spp='256' \
        --outroot="$ROOT"/data/scenes/dragon_specular_imh512_uvs512_spp256/ \
        --tmpdir="$ROOT"/tmp/
    ```
   Any Python can be used for this step, not necessarily the Blender-bundled
   Python, because there is no Blender-specific operation.

1. Distribute the rendering jobs to your render farm, depending on your
   infrastructure.

1. Glob the rendered data and dump the file list to disk, such that the
   training pipeline can just load this tiny file and know immediately which
   camera-light configuration has missing data (caused by, e.g., failed
   rendering jobs):
    ```
    python "$ROOT"/sim/data_gen/gen_file_stats.py \
        --data_root="$ROOT"'/data/scenes/dragon_specular_imh512_uvs512_spp256/' \
        --out_json="$ROOT"/data/scenes/dragon_specular_imh512_uvs512_spp256.json
    ```

### Tips

* Because UV unwrapping is dependent only on the object's geometry and
  independent of the camera-light configuration, you only need to perform it
  once per scene. Given that it might also be an expensive operation (e.g., if
  your model has a high poly. count), consider caching the UV unwrapping results
  to the disk, and then letting each rendering job just load them using
  `--cached_uv_unwrap` in `render.py`.
