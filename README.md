# VAE Disentanglement
Use a product space constructed from simple 1D structure to do disentanglement learning. 

The hypothesis for this work is that a disentangled representation can be specified as $n$ unique 1d factors that are *independent*. Given this hypothesis, there are a few different 1d topologies that might be relevent for common datasets.

1. Disjoint set of points
2. Line segment with end-points
3. Loop, or line segment with periodic boundary conditions

Considering the [3dshape](https://github.com/deepmind/3d-shapes), we can see these patterns naturally emerge. For example, there are a disjoint set of shapes: sphere, cube, cylinder etc. The rotation angle of the camera is a loop, assuming that the angle is rotated about the full scene. The object size is a line segment with end points, given that the sizes have min and max values.

As a first task, we can specify the desired set of latent variables and topologies which naturally form a product space. Perhaps later, we can investigate automatic discovery of these latent variables...

## Low-Enenergy Surfaces
Consider the [tilted prior](https://openreview.net/forum?id=YlGsTZODyjz). Here, we can see that the surface of a sphere (with radius controlled by $\tau$), has the lowest possible KLD or "energy" in the latent space. The KLD or "energy" increases quadratically with the distance to the low-energy manifold.

We can generalize this notation to fit our proposed simple 1d topologies. For example, the disjoint set of points can be represented as a set of delta functions, where the KLD / "energy" would correspond to a mixture of Gaussians. We can deal with the loop by mapping it to a circle in $\R^2$, where the "energy" is described by the corresponding tilted distribution.


## Model
Initially, we selected the architecture from the [NVAE](https://arxiv.org/abs/2007.03898) paper. Block diagrams of the structure are shown below:  

<p align="center">
<img src="imgs/nvae.png" width=700>
</p>

For some unknown reason the squeeze and excitation blocks prevent proper learning to occur? Overall, things aren't quite working very well. We then move on...

A stripped down version of the VQ-VAE from the original [stable diffusion](https://github.com/CompVis/stable-diffusion) is used instead. This worked much better...

## Dataset
Initial testing is being performed using the [3dshape](https://github.com/deepmind/3d-shapes) dataset from deepmind. An immediate problem is that this dataset isn't general enough for the paper we're trying to write (the only latent variable types are discrete or bounded linear). We require a dataset with circular latent variables. 

## Basic Results
Training a plain VAE is working better. This is a results after maybe 30 min of training (ie. not convergence). Top row is input, bottom is reconstructed.

<p align="center">
<img src="imgs/3dshapes.png" width=700>
</p>

Some difficulty is encountered with the general disentanglement goal. The current thought process is that some form of annealing may work best (ie. the "energy" begins as flat, then over time becomes more sharp to force encoded points onto the low-energy manifold).