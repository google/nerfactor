# NeRFactor: Data Generation

This folder contains code and instructions for:
1. rendering images of a scene and subsequently converting these image data into
   a TensorFlow dataset (`nerf_synth/`),
1. processing the two real 360-degree captures from NeRF (`nerf_real/`), and
1. converting the MERL binary BRDFs into a TensorFlow dataset (`merl/`).


## Converting the MERL Binary BRDFs Into a TensorFlow Dataset

1. Download the MERL BRDF dataset:
    ```bash
    wget
    ```

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

Go to [`nerf_synth/`](./nerf_synth) and follow the instructions there.


## NeRF: Real Captures

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


## Philip et al. 2019: Real Captures

1. Download a real capture from
   [here](https://repo-sam.inria.fr/fungraph/deep-relighting/index.html).

1. Convert these real images and [Bundler](https://github.com/snavely/bundler_sfm#output-format)
   poses into our format:
    ```bash
    scene='stonehenge'
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    scene_dir="$proj_root/data/philip2019multi/$scene"
    h='256'
    n_vali='2'
    if [[ "$scene" == stonehenge ]]; then
        exclude='0,1,2,3,4,141,142,143,144,145'
    else
        exclude=''
    fi
    outroot="$proj_root/data/philip2019multi_proc/${scene}"

    REPO_DIR="$repo_dir" "$repo_dir/data_gen/philip2019multi_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --exclude="$exclude" --outroot="$outroot"
    ```
