# NeRFactor: Model

This folder is a TensorFlow 2 (eager) pipeline for training, validation, and
testing.

`trainvali.py` and `test.py` are the main scripts with the
training/validation/testing loops. `config/` contains the example configuration
files, from which the pipeline parses arguments. The full NeRFactor model is
specified in `models/nerfactor.py`, and the main data loader is specified in
`datasets/nerf_shape.py`.

Given multi-view, posed images of the scene and the MERL BRDF dataset, we (1)
first learn data-drive BRDF priors, (2) distill NeRF's (noisy) geometry so
that we can refine it, and finally (3) jointly optimize for the shape,
reflectance, and illumination.


## Visualization

Different models in this pipeline may produce visualization in their own ways,
but for all of them, you can visualize the training and validation losses with:
```bash
tensorboard --logdir="$outroot" --bind_all
```
and find handy links to the visualization webpages under the "TEXT" tab
in TensorBoard. For testing, the link to the compiled video will be printed at
the end of the run.


## Preparation

1. (Only once for all scenes) Learn data-driven BRDF priors (using a single
   GPU suffices):
    ```bash
    gpus='0'

    # I. Learning BRDF Priors (training and validation)
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    data_root="$proj_root/data/brdf_merl_npz/ims512_envmaph16_spp1"
    outroot="$proj_root/output/train/merl"
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='brdf.ini' --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"

    # II. Exploring the Learned Space (validation and testing)
    ckpt="$outroot/lr1e-2/checkpoints/ckpt-50"
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/explore_brdf_space_run.sh" "$gpus" --ckpt="$ckpt"
    ```

1. Train a vanilla NeRF, optionally using multiple GPUs:
    ```bash
    scene='hotdog_2163'
    gpus='0,1,2,3'
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''
    data_root="$proj_root/data/selected/$scene"
    if [[ "$scene" == chichen || "$scene" == russian_church || "$scene" == stonehenge ]]; then
        imh='256'
    else
        imh='512'
    fi
    if [[ "$scene" == pinecone* || "$scene" == vasedeck* || "$scene" == chichen || "$scene" == russian_church || "$scene" == stonehenge ]]; then
        near='0.1'; far='2'
    else
        near='2'; far='6'
    fi
    if [[ "$scene" == ficus* || "$scene" == hotdog_probe_16-00_latlongmap ]]; then
        lr='1e-4'
    else
        lr='5e-4'
    fi
    outroot="$proj_root/output/train/${scene}_nerf"
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"
    ```
   Check the quality of this NeRF geometry by inspecting the visualization HTML
   for the alpha and normal maps. You might need to re-run this with another
   learning rate if the estimated NeRF geometry is too off.

1. Compute geometry buffers for all views by querying the trained NeRF:
    ```bash
    scene='hotdog_2163'
    gpus='0'
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    viewer_prefix='http://vision38.csail.mit.edu' # or just use ''
    data_root="$proj_root/data/selected/$scene"
    if [[ "$scene" == chichen || "$scene" == russian_church || "$scene" == stonehenge ]]; then
        imh='256'
    else
        imh='512'
    fi
    if [[ "$scene" == ficus* || "$scene" == hotdog_probe_16-00_latlongmap ]]; then
        lr='1e-4'
    else
        lr='5e-4'
    fi
    trained_nerf="$proj_root/output/train/${scene}_nerf/lr$lr"
    occu_thres='0.5'
    if [[ "$scene" == pinecone* || "$scene" == stonehenge || "$scene" == russian_church ]]; then
        scene_bbox='-0.3,0.3,-0.3,0.3,-0.3,0.3'
    elif [[ "$scene" == chichen ]]; then
        scene_bbox='-0.5,0.5,-0.5,0.5,-0.5,0.5'
    elif [[ "$scene" == vasedeck* ]]; then
        scene_bbox='-0.2,0.2,-0.4,0.4,-0.5,0.5'
    else
        scene_bbox=''
    fi
    out_root="$proj_root/output/surf/$scene"
    mlp_chunk='375000' # bump this up until GPU gets OOM for faster computation
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/geometry_from_nerf_run.sh" "$gpus" --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk"
    ```


## Training, Validation, and Testing

Pre-train geometry MLPs (pre-training), jointly optimize shape, reflectance,
and illumination (training and validation), and finally perform simultaneous
relighting and view synthesis results (testing):
```bash
scene='hotdog_2163'
gpus='0,1,2,3'
overwrite='True'
proj_root='/data/vision/billf/intrinsic/sim'
repo_dir="$proj_root/code/nerfactor"
viewer_prefix='http://vision38.csail.mit.edu' # or just use ''

# I. Shape Pre-Training
data_root="$proj_root/data/selected/$scene"
imh='512'
if [[ "$scene" == pinecone* || "$scene" == vasedeck* ]]; then
    near='0.1'; far='2'; use_nerf_alpha=true
else
    near='2'; far='6'; use_nerf_alpha=false
fi
surf_root="$proj_root/output/surf/$scene"
shape_outdir="$proj_root/output/train/${scene}_shape"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

# II. Joint Optimization (training and validation)
shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
brdf_ckpt="$proj_root/output/train/merl/lr1e-2/checkpoints/ckpt-50"
test_envmap_dir="$proj_root/data/envmaps/for-render_h16/test"
outroot="$proj_root/output/train/${scene}_nerfactor"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerfactor.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,test_envmap_dir=$test_envmap_dir,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

# III. Simultaneous Relighting and View Synthesis (testing)
ckpt="$outroot/lr1e-3/checkpoints/ckpt-10"
if [[ "$scene" == pinecone* || "$scene" == vasedeck* ]]; then
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt"
else
    REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo
fi
```

Training and validation (II) will produce an HTML of the factorization results:
normals, visibility, albedo, reflectance, and re-rendering. Testing (III) will
produce a video visualization of the scene as viewed from novel views and relit
under novel lighting conditions.

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
