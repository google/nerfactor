## Relightable Neural Renderer (RNR): Synthetic Data

TODO This section is relevant only if you want to process your own capture. If our
processed version of Philip et al.'s data already suffices for
your purpose, it is much easier to just
[download our processed version](https://github.com/google/nerfactor#data)
and skip the following instructions.

1. Download a real capture from
   [here]()
   and unzip it to `$proj_root/data/rnr/material_sphere`.

1. Convert these data and camera poses into our format:
    ```bash
    proj_root='/data/vision/billf/intrinsic/sim'
    repo_dir="$proj_root/code/nerfactor"
    scene_dir="$proj_root/data/rnr/material_sphere"
    their_quan_results_dir="$proj_root/data/rnr/material_sphere_results_quantitative"
    their_qual_results_dir="$proj_root/data/rnr/material_sphere_results"
    h='512'
    n_vali='2'
    outroot="$proj_root/data/rnr_proc/material_sphere"
    REPO_DIR="$repo_dir" "$repo_dir/data_gen/rnr_synth/make_dataset_run.sh" --scene_dir="$scene_dir" --their_quan_results_dir="$their_quan_results_dir" --their_qual_results_dir="$their_qual_results_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"
    ```


## Philip et al. 2019: Real Captures

This section is relevant only if you want to process your own capture. If our
processed version of Philip et al.'s data already suffices for
your purpose, it is much easier to just
[download our processed version](https://github.com/google/nerfactor#data)
and skip the following instructions.

1. Download a real capture from
   [here](https://repo-sam.inria.fr/fungraph/deep-relighting/index.html)
   and unzip it to `$proj_root/data/philip2019multi/$scene`.

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
