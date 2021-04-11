# NeRFactor: Data Generation

This folder contains code and instructions for:
1. rendering images of a scene and subsequently converting these image data into
   a TensorFlow dataset (`synth/`),
1. processing the two real 360-degree captures from NeRF (`real/`), and
1. converting the MERL binary BRDFs into a TensorFlow dataset (`merl/`).


## Synthetic Data

TODO


## Real Captures

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

    REPO_DIR="$repo_dir" "$repo_dir/data_gen/real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"
    ```


## Converting the MERL Binary BRDFs Into a TensorFlow Dataset

1. Go to the correct folder for this task:
    ```bash
    cd merl
    ```

1. Start the conversion:
    ```bash
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    merl_dir='/data/vision/billf/intrinsic/sim/data/brdf_merl'
    out_dir='/data/vision/billf/intrinsic/sim/data/brdf_merl_npz/ims512_envmaph16_spp1'

    REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$merl_dir" "$out_dir"
    ```
