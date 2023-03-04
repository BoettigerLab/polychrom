# polychrom

This code is used for polymer simulations of chromatin loop extrusion and heterotypic sticky interactions.  

This repository is the Boettiger lab fork of the "polychrom" repository developed by the Open Chromatin Collective (Open2C), github.com/open2c.  Most of the framework was written by Max Imakaev and colleagues, and it builds off work completed and published in the Mirny lab using its predecessor, openmm-polymer.  The project is largely a set of python wrappers for chromatin polymer simulations, built on top of the [openmm system](http://docs.openmm.org/latest/userguide/application.html#installing-openmm) for molecular dynamics simulations.  Additional documentation can be found at the [github.com/open2c/polychrom](github.com/open2c/polychrom)  Our version of the repository is built on top of the following polychrom release:

[![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

## Some additions
Notable additions include:
- new functions to allow targeted loading of cohesin. We used these to explore potential explanations of the unique 3D organization of the Sox9 locus, see our [Sox9 project](github.com/BoettigerLab/sox9-ORCA-2022) repository for more details. 

- new functions to allow the nature of sticky interactions among monomers to evolve over time, as a consequence of the 3D interactions among the monomers. We use this to explore the effect of 3D structure on the maintenance and spreading of Polycomb-chromatin states in our [Polycomb project](github.com/BoettigerLab/Polycomb-ORCA-2022)

## Install instructions
polychrom runs off the openmm system for molecular dynamics simulation. Official installation instructions for openmm are here: 
http://docs.openmm.org/latest/userguide/application.html#installing-openmm
If you have Anaconda or one of its derivatives like Miniconda already installed, and your GPU drivers are also installed (or you intend to use CPU), installation should be a simple: 
`conda install -c conda-forge openmm`

Once you openmm, there are a few other python dependencies you may need. Most of these are listed in the 'requirements.txt'. Python will also tell you if any packages are missing and you can simply pip install them then. 

If you do plan to use a CUDA GPU, you will need to make sure your openmm install specifies the matching cudatoolkit.
An NVIDIA "Titan X Pascal" or "Titan V" are currently the benchmark graphics cards.  

