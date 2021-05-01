# NeRFactor: Rendering NeRF-Like Data

TODO
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

1. "Install" Blender-Python (the binaries are pre-built, so just download
   and unzip):
    ```bash
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    # Make directory
    mkdir "$proj_root"/software
    cd "$proj_root"/software
    # Download
    wget https://download.blender.org/release/Blender2.83/blender-2.83.4-linux64.tar.xz
    # Unzip the pre-built binaries
    tar -xvf blender-2.83.4-linux64.tar.xz
    ```

1. Install the dependencies to this *Blender-bundled* Python:
    ```bash
    cd blender-2.83.4-linux64/2.83/python/bin
    # Install pip for THIS Blender-bundled Python
    curl https://bootstrap.pypa.io/get-pip.py | ./python3.7m
    # If the above fails, make sure you deactivate your Conda environment
    # Use THIS pip to install other dependencies
    ./pip install absl-py tqdm ipython numpy Pillow opencv-python
    ```


## Rendering

```bash
scene='hotdog'
light='2188'
proj_root='/data/vision/billf/intrinsic/sim'
blender_bin="$proj_root/software/blender-2.83.4-linux64/blender"
repo_dir="$proj_root/code/nerfactor"
scene_path="$proj_root/data/scenes/$scene.blend"
light_path="$proj_root/data/envmaps/for-render_h16/train/$light.hdr"
cam_dir="$proj_root/data/cams/nerf"
test_light_dir="$proj_root/data/envmaps/for-render_h16/test"
light_inten='3'
outdir="$proj_root/data/render_outdoor_inten${light_inten}_gi/${scene}_${light}"
REPO_DIR="$repo_dir" BLENDER_BIN="$blender_bin" "$repo_dir/data_gen/nerf_synth/render_run.sh" --scene_path="$scene_path" --light_path="$light_path" --cam_dir="$cam_dir" --test_light_dir="$test_light_dir" --light_inten="$light_inten" --outdir="$outdir"
```

We modify our parallel rendering code to this current version that renders all
views sequentially for portability. If you have a cluster, consider distributing
all views across the cluster to render them in parallel. The heavylifting
function to be distributed is `render_view()` in `render.py`.
