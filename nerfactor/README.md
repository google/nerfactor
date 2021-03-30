# NeRFactor: Main Model

This folder is a TensorFlow 2 (eager) pipeline for training, validation, and
testing.

`trainvali.py` and `test.py` are the main scripts with the
training/validation/testing loops. `config/` contains the example configuration
files, from which the pipeline parses arguments. The full NeRFactor model is
specified in `models/nerfactor.py`, and the data loader is specified in
`datasets/nerf_shape.py`.

Each model in this pipeline has its own training/validation visualization,
but for all of them, you can visualize the losses with:
```
tensorboard --logdir="$outroot" --bind_all
```
and find handy links to the visualization webpages under the "TEXT" tab
in TensorBoard.


## Training

1. (Only once for all scenes) Learn data-driven BRDF priors (using a single
   GPU suffices):
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/brdf_merl_npz/ims512_envmaph16_spp1'
    outroot='/data/vision/billf/intrinsic/sim/output/train/merl'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0' --config='brdf.ini' \
        --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

1. Train a vanilla NeRF, optionally using multiple GPUs:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    near='2' # use '0.1' if real 360 data
    far='6' # use '2' if real 360 data
    outroot='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='nerf.ini' \
        --config_override="data_root=$data_root,near=$near,far=$far,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

1. Compute geometry buffers for all views by querying the trained NeRF:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    trained_nerf='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163/lr5e-4'
    out_root='/data/vision/billf/intrinsic/sim/output/surf/hotdog_2163'
    imh='256'
    scene_bbox='' # useful for bounding real scenes: '-0.2,0.2,-0.4,0.4,-0.5,0.3' for vasedeck; '' for pinecone
    occu_thres='0.3' # useful for removing floaters in real scenes
    mlp_chunk='375000' # bump this up until GPU gets OOM for faster computation

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/geometry_from_nerf_run.sh '0' \
        --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" \
        --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk"
    ```

1. Pre-train geometry MLPs that cache the NeRF geometry, which takes only around
   20 minutes on four GPUs:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    near='2' # use '0.1' if real 360 data
    far='6' # use '2' if real 360 data
    surf_root='/data/vision/billf/intrinsic/sim/output/surf/hotdog_2163'
    outroot='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163_shape'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='shape.ini' \
        --config_override="data_root=$data_root,near=$near,far=$far,data_nerf_root=$surf_root,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

1. Jointly optimize shape, reflectance, and illumination:
    ```
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    near='2' # use '0.1' if real 360 data
    far='6' # use '2' if real 360 data
    surf_root='/data/vision/billf/intrinsic/sim/output/surf/hotdog_2163'
    shape_ckpt='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163_shape/lr1e-2/checkpoints/ckpt-2'
    brdf_ckpt='/data/vision/billf/intrinsic/sim/output/train/merl/lr1e-2/checkpoints/ckpt-50'
    test_envmap_dir='/data/vision/billf/intrinsic/sim/data/envmaps/for-render_h16/test'
    outroot='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163_nerfactor'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='nerfactor.ini' \
        --config_override="data_root=$data_root,near=$near,far=$far,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,test_envmap_dir=$test_envmap_dir,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

### Tips

* Even if eager execution is used throughout, the data pipeline still runs in
  the graph mode (by design). This may complicate debugging the data pipeline
  in that you may not be able to insert breakpoints. Take a look at
  `debug/debug_dataloader`, where we call the dataset functions outside of
  the pipeline, insert breakpoints there, and debug.
* For easier and faster debugging, consider turning on `debug`. For instance,
  with this flag on, `trainvali.py` will NOT decorate the main training step
  with `@tf.function` (easier) and only load a single datapoint (faster).


## Testing

Run the testing pipeline:
```
CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/nerfactor/nerfactor/test.py \
    --ckpt="$ROOT"'/output/train/lr:1e-3_mgm:-1/checkpoints/ckpt-43'
```
which runs inference with the given checkpoint on all test data, and eventually
produces a video visualization whose frames correspond to different camera-light
configurations.

