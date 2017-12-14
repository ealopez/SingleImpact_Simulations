
This repository contains information and python code to perform computational simulations of an atomic force microscope (AFM) probe interacting
with a viscoelastic surface. The cantilever (probe) is excited with an impulsive excitation. Specifically, it is applied a sinus cardinalis (sinc) force
to the tip and the cantilever is allowed to oscillate over a viscoelastic surface.
The details of the excitation and the modeled material can be found in:

- LÛpez-Guerra, E.A.; Banfi, F.; Solares, S.D.; Ferrini, G.;"Theory of single-impact atomic force spectroscopy in liquids with material contrast," submitted.


This repository contains the following core files:

- AFM_sinc.py  :   This is a python library containing relevant information related to sinc excitation simulations performed over a viscoelastic surface.
		   The core function used to obtain the data reported in the previously mentioned publication is "MDR_SLS_sinc". This function contains an
                   implementation of the Method of Dimensionality Reduction (Popov, V. L., & Heﬂ, M. (2015). Method of dimensionality reduction in contact mechanics and friction. Springer Berlin Heidelberg.)
                   This method of dimensionality reduction (MDR) allows to solve numerically the boundary value problem of an indenter penetrating a viscoelastic halfspace.
                   In this case we deal with the case of a parabolic indenter. The MDR method is alligned with Ting's theory of viscoelastic indentation (Ting, T. C. T. (1966, December). The contact stresses between a rigid indenter and a viscoelastic half-space. ASME.)
                   MDR allows to address Ting's challenging formulation that is applicable for arbitrary loading histories.
		   We also provide as a reference Lee and Radok formulation of viscoelastic indentation (Lee, E. H., & Radok, J. R. M. (1960). The contact problem for viscoelastic bodies. Journal of Applied Mechanics, 27(3), 438-444.)
   		   Lee and Radok formulation is significantly simpler than Ting's method but is only (strictly) applicable for cases of monotonic loading (monotonic increase of contact radius).

- Sinc_excitation_simulation.ipynb : This is a jupyter (ipython) notebook containing the simulations which are performed in the basis of the functions contained in AFM_sinc.py


What you will need to be able to run the code and read the jupyter notebook:

- Python 2.7 : we suggest to download the Anaconda distribution (https://www.anaconda.com/download/)

- You will also need the "numba" package, an optimizing compiler that we use to make the code run faster, this probably comes with the anaconda distribution, if not it is very easy to add it once you have the anaconda distribution.

- You will also need the jupyter notebook which may be downloaded along the anaconda distribution.