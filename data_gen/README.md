# NeRFactor: Data Generation

This folder contains code and instructions for:
1. rendering images of a scene and subsequently converting these image data into
   a TensorFlow dataset (`nerf_synth/`),
1. processing the two real 360-degree captures from NeRF (`nerf_real/`), and
1. converting the MERL binary BRDFs into a TensorFlow dataset (`merl/`).


## Converting the MERL Binary BRDFs Into a TensorFlow Dataset

1. Download
   [the MERL BRDF dataset](https://cdfg.csail.mit.edu/wojciech/brdfdatabase)
   to `$proj_root/data/brdf_merl/`.

1. Convert the dataset into our format:
    ```bash
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    indir="$proj_root/data/brdf_merl"
    ims='256'
    outdir="$proj_root/data/brdf_merl_npz/ims${ims}_envmaph16_spp1"
    REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"
    ```
   In this conversion process, the BRDFs are visualized to `$outdir/vis`,
   in the forms of characteristic clices and renders.


## NeRF: Synthetic Data

This section is relevant only if you want to render your own data, e.g., using
your own scene or light probe. If our data suffice for your purpose already, it
is much easier to just
[download our rendering](https://github.com/google/nerfactor#data)
and skip the following instructions.

Go to [`nerf_synth/`](./nerf_synth) and follow the instructions there.


## NeRF: Real Captures

This section is relevant only if you want to process your own capture. If our
processed version of the NeRF 360-degree real captures already suffices for
your purpose, it is much easier to just
[download our processed version](https://github.com/google/nerfactor#data)
and skip the following instructions.

1. Download the 360-degree real captures by NeRF from
   [here](https://drive.google.com/file/d/1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB/view?usp=sharing).

1. Convert these real images and COLMAP poses into our format:
    ```bash
    scene='pinecone'
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    scene_dir="$proj_root/data/nerf_real_360/$scene"
    h='512'
    n_vali='2'
    outroot="$proj_root/data/nerf_real_360_proc/${scene}"
    REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"
    ```
