# NeRFactor

[[Paper]]()
[[Video]](https://www.youtube.com/watch?v=UUVSPJlwhPg)
[[Project]](http://people.csail.mit.edu/xiuming/projects/nerfactor/)

![teaser](assets/teaser.jpg)

This is the authors' code release for:
> **NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination**  
> Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul Debevec, William T. Freeman, Jonathan T. Barron  
> **arXiv**

This is not an officially supported Google product.


## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/google/nerfactor.git
    ```

1. Install a Conda environment with all dependencies:
    ```bash
    cd nerfactor
    conda env create -f environment.yml
    conda activate nerfactor
    ```

Tips:
* You can find the TensorFlow version, cuDNN version, and CUDA version in
  `environment.yml`.
* The IPython dependency in `environment.yml` is for `IPython.embed()` alone.
  If you are not using that to insert breakpoints during debugging, you can
  take it out (it should not hurt to just leave it there).


## Data

If you are using our data, see "Downloads" on the
[project page](http://people.csail.mit.edu/xiuming/projects/nerfactor/).

If you are BYOD (bringing your own data), go to [`data_gen/`](./data_gen) to
either render your own synthetic data or process your real captures.


## Running the Code

Go to [`nerfactor/`](./nerfactor) and follow the instructions there.


## Issues or Questions?

If the issue is code-related, please open an issue here.

For questions, please also consider opening an issue as it may benefit future
reader. Otherwise, email [Xiuming Zhang](http://people.csail.mit.edu/xiuming).


## Changelog

* 06/01/2021: Initial release.
