# NeRFactor

[[Project]](http://nerfactor.csail.mit.edu/)

![teaser](assets/teaser.png)

This is the authors' code release for:
> **Neural Factorization of Shape and Reflectance under an Unknown Illumination**  
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

### Tips

* The IPython dependency in `environment.yml` is for `IPython.embed()` alone.
  If you are not using that to insert breakpoints during debugging, you can
  take it out (it should not hurt to just leave it there).


## Issues or Questions?

If the issue is code-related, please open an issue here.

For questions, please also consider opening an issue as it may benefit future
reader. Otherwise, email [Xiuming Zhang](http://people.csail.mit.edu/xiuming).


## Changelog

* TODO/TODO/2021: Initial release.
