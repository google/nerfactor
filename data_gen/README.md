# NeRFactor: Data Generation

This folder contains code and instructions for:
1. rendering images of a scene and subsequently converting these image data into
   a TensorFlow dataset,
1. processing the two real 360-degree captures from NeRF, and
1. converting the MERL binary BRDFs into a TensorFlow dataset.


## Rendering Images and Making a TensorFlow Dataset

TODO


## Processing Real Captures Into a Dataset

1. Download the 360-degree real captures by NeRF from
   [here](https://drive.google.com/file/d/1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB/view?usp=sharing).

1. Convert these real images and COLMAP poses into our format:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    scene_dir='/data/vision/billf/intrinsic/sim/data/nerf_real_360/pinecone'
    h='512'
    n_vali='2'
    outroot='/data/vision/billf/intrinsic/sim/data/nerf_real_360_proc/pinecone'

    REPO_DIR="$repo_dir" "$repo_dir"/data_gen/real/make_dataset_run.sh \
        "$scene_dir" "$h" "$n_vali" "$outroot"
    ```


## Converting the MERL Binary BRDFs Into a TensorFlow Dataset

1. Go to the correct folder for this task:
    ```
    cd merl
    ```

1. Start the conversion:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    merl_dir='/data/vision/billf/intrinsic/sim/data/brdf_merl'
    out_dir='/data/vision/billf/intrinsic/sim/data/brdf_merl_npz/ims512_envmaph16_spp1'

    REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$merl_dir" "$out_dir"
    ```
