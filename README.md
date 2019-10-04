# Faster PET Reconstruction with Non-Smooth Priors by Randomization and Preconditioning

This repository code to reproduce the results in [1] which proposes the use of the <b>Stochastic Primal-Dual Hybrid Gradient algorithm (SPDHG)</b> [2] for PET imaging with non-smooth priors (see also [3]). The results are on real clinical PET data in span 1 from the Siemens Biograph mMR based at UCLH Macmillan Cancer Centre. SPDHG is a direct generalization of the popular Primal-Dual Hybrid Gradient algorithm (PDHG) also known as the Chambolle-Pock algorithm [4, 5].

[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schönlieb, “Faster PET reconstruction with non-smooth priors by randomization and preconditioning,” Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07

Example results for various priors on FDG data are shown below.

<p align="center"><img src="https://github.com/mehrhardt/spdhg_pet/blob/master/example_fdg.png" width="70%" border="0"/></p>


## Installation of dependencies

You need to install [ODL](https://github.com/odlgroup/odl) and [NiPET](https://github.com/pjmark/NIPET) [6] to run the examples.

We recommend to use conda to manage the correct versions of the packages. If you want to use conda, you could start like
```
conda create --name pet python=2.7
source activate pet
```

Then get some packages that nipet [NiPET](https://github.com/pjmark/NIPET) need.
```
pip2 install numpy dicom
```

### Get PET data

Download the PET data set available [here](https://doi.org/10.5281/ZENODO.1472951), e.g.
```
wget https://zenodo.org/record/1472951/files/amyloidPET_FBP_TP0.zip path_to_data
```

### NiPET

Clone the GIT repository and install (use `path_to_data/umap` when asked for the attenuation maps)
```
git clone https://github.com/pjmark/NIPET.git path_to_nipet
cd path_to_nipet
pip2 install --no-binary :all: --verbose .
```

### (most recent) [ODL](https://github.com/odlgroup/odl)

Clone the GIT repository and install
```
git clone https://github.com/odlgroup/odl.git path_to_odl
cd path_to_odl
pip2 install -e .
```

The code might also be compatible with older versions of [ODL](https://github.com/odlgroup/odl) which are slightly easier to install:
```
pip2 install odl
```

## Examples

Assuming you have installed the dependencies as below, you can run various examples as outlined below.

The numerical examples in [1] can be reproduced with 

* [Maximum Likelihood (ML) example](python/ml.py)
* [ML example which breaks OSEM](python/ml_bin.py)
* [Maximum A-Posteriori (MAP) example with (isotropic) total variation (TV) prior](python/map_tv.py)
* [MAP example with anisotropic TV prior](python/map_atv.py)
* [MAP example with directional TV prior (uses MRI)](python/map_dtv.py)
* [MAP example with total generalized variation (TGV) prior](python/map_tgv.py)

and the figures be recreated with

* [Print Figure 2 and side information for Figures 10 and 11](python/print_param_and_sideinfo.py)
* [Print Figures 1, 3-11](python/print_figs.py)


## References

[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schönlieb, “Faster PET reconstruction with non-smooth priors by randomization and preconditioning,” Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07

[2] A. Chambolle, M. J. Ehrhardt, P. Richtárik, and C.-B. Schönlieb, “Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and Imaging Applications,” SIAM J. Optim., vol. 28, no. 4, pp. 2783–2808, 2018.

[3] M. J. Ehrhardt, P. J. Markiewicz, P. Richtárik, J. Schott, A. Chambolle, and C.-B. Schönlieb, “Faster PET Reconstruction with a Stochastic Primal-Dual Hybrid Gradient Method,” in Proceedings of SPIE, 2017, vol. 10394, pp. 1–12.

[4] A. Chambolle and T. Pock, “A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging,” J. Math. Imaging Vis., vol. 40, no. 1, pp. 120–145, Dec. 2011.

[5] T. Pock and A. Chambolle, “Diagonal Preconditioning for First Order Primal-Dual Algorithms in Convex Optimization,” in Proceedings of the IEEE International Conference on Computer Vision, 2011, pp. 1762–1769.

[6] P. J. Markiewicz et al., “NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy and Precision PET Imaging and Analysis,” Neuroinformatics, vol. 16, no. 1, pp. 95–115, 2018.