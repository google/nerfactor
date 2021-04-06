# NeRFactor: Generate Real Datasets

```
repo_dir='/data/vision/billf/intrinsic/sim/code/nerfactor'
scene_dir='/data/vision/billf/intrinsic/sim/data/nerf_real_360/pinecone'
h='512'
n_vali='2'
outroot='/data/vision/billf/intrinsic/sim/data/nerf_real_360_proc/pinecone_new'

REPO_DIR="$repo_dir" "$repo_dir"/data_gen/real/make_dataset_run.sh \
    --scene_dir="$scene_dir" \
    --h="$h" \
    --n_vali="$n_vali" \
    --outroot="$outroot"
```
