# NeRFactor: Model

This folder is a TensorFlow 2 (eager) pipeline for training, validation, and
testing.

`trainvali.py` and `test.py` are the main scripts with the
training/validation/testing loops. `config/` contains the example configuration
files, from which the pipeline parses arguments. The full NeRFactor model is
specified in `models/nerfactor.py`, and the main data loader is specified in
`datasets/nerf_shape.py`.

Different model in this pipeline may produce visualization in their own ways,
but for all of them, you can visualize the training and validation losses with:
```bash
tensorboard --logdir="$outroot" --bind_all
```
and find handy links to the visualization webpages under the "TEXT" tab
in TensorBoard. For testing, the link to the compiled video will be printed at
the end of the run.


## Training and Validation

Given multi-view, posed images of the scene and the MERL BRDF dataset, we (1)
first learn data-drive BRDF priors, (*2) distill NeRF's (noisy) geometry so
that we can refine it, and finally (3) jointly optimize for the shape,
reflectance, and illumination.

1. (Only once for all scenes) Learn data-driven BRDF priors (using a single
   GPU suffices):
    ```bash
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/brdf_merl_npz/ims512_envmaph16_spp1'
    outroot='/data/vision/billf/intrinsic/sim/output/train/merl'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0' --config='brdf.ini' \
        --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

1. Train a vanilla NeRF, optionally using multiple GPUs:
    ```bash
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    imh='512'
    near='2' # use '0.1' if real 360 data
    far='6' # use '2' if real 360 data
    outroot='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163'
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='nerf.ini' \
        --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```

1. Compute geometry buffers for all views by querying the trained NeRF:
    ```bash
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root='/data/vision/billf/intrinsic/sim/data/render_outdoor_inten3_gi/hotdog_2163'
    trained_nerf='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163/lr5e-4'
    out_root='/data/vision/billf/intrinsic/sim/output/surf/hotdog_2163'
    imh='512'
    scene_bbox='' # '' for synthetic scenes, '-0.2,0.2,-0.4,0.4,-0.5,0.3' for vasedeck, and '-0.3,0.3,-0.3,0.3,-0.3,0.3' for pinecone
    occu_thres='0' # '0' for synthetic scenes, and '0.5' for real scenes
    mlp_chunk='375000' # bump this up until GPU gets OOM for faster computation

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/geometry_from_nerf_run.sh '0' \
        --data_root="$data_root" --trained_nerf="$trained_nerf" \
        --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" \
        --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk"
    ```

1. Pre-train geometry MLPs (distilling the NeRF geometry) and then jointly
   optimize shape, reflectance, and illumination:
    ```bash
    scene='hotdog_2163'
    repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
    data_root="/data/vision/billf/intrinsic/sim/data/selected/$scene"
    imh='512'
    if [[ "$scene" == pinecone* || "$scene" == vasedeck* ]]; then
        near='0.1'; far='2'; use_nerf_alpha=true
    else
        near='2'; far='6'; use_nerf_alpha=false
    fi
    surf_root="/data/vision/billf/intrinsic/sim/output/surf/$scene"
    shape_outdir="/data/vision/billf/intrinsic/sim/output/train/${scene}_shape"
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''
    shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
    brdf_ckpt='/data/vision/billf/intrinsic/sim/output/train/merl/lr1e-2/checkpoints/ckpt-50'
    test_envmap_dir='/data/vision/billf/intrinsic/sim/data/envmaps/for-render_h16/test'
    outroot="/data/vision/billf/intrinsic/sim/output/train/${scene}_nerfactor"

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix"

    REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/trainvali_run.sh '0,1,2,3' --config='nerfactor.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,test_envmap_dir=$test_envmap_dir,outroot=$outroot,viewer_prefix=$viewer_prefix"

    ```

### Tips

* Even if eager execution is used throughout, the data pipeline still runs in
  the graph mode (by design). This may complicate debugging the data pipeline
  in that you may not be able to insert breakpoints. Take a look at
  `debug/debug_dataloader`, where we call the dataset functions outside of
  the pipeline, insert breakpoints there, and debug.
* For easier and faster debugging, consider turning on `debug`. For instance,
  with this flag on, `trainvali.py` will NOT decorate the main training step
  with `@tf.function` (easier) and only load a single datapoint each epoch
  (faster).


## Testing: View Synthesis, Relighting, and Both Simultaneously

After the factorization, one can synthesize novel views under lighting
conditions by testing the trained model:
```bash
repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
ckpt='/data/vision/billf/intrinsic/sim/output/train/hotdog_2163_nerfactor/lr1e-3/checkpoints/ckpt-10'

# Remove `--color_correct_albedo` for real scenes (no validation albedo)
REPO_DIR="$repo_dir" "$repo_dir"/nerfactor/test_run.sh '0' --ckpt="$ckpt" --color_correct_albedo
```
This eventually produces a video visualization of the scene as viewed from novel
views and relit under novel lighting conditions.

