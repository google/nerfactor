# NeRFactor: Third-Party Code

This folder contains third-party code. These functions are called by other
functions, and you will NOT need to run these directly.


## MERL Utility Functions

This code was adapted from [the code release](https://brdf.compute.dtu.dk/) of:
> **On Optimal, Minimal BRDF Sampling for Reflectance Acquisition**  
> Jannik Boll Nielsen, Henrik Wann Jensen, Ravi Ramamoorthi  
> **TOG 2015**

The changes are minor:
1. Updating the `print` statements according to the Python 3 syntax;
1. Fixing a typo in the comment section of `readMERLBRDF.py`.


## xiuminglib

This is a release of [xiuminglib](https://xiuminglib.readthedocs.io/en/latest/)
that was used in NeRFactor. Make sure to use this release rather than the head
of xiuminglib, which may have gone ahead and no longer be compatible with this
project.
