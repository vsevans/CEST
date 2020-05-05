# CEST
Codes and materials for CEST MRI data analysis

The Matlab script (CEST_Script.m), functions and data contained here-in are provided in support of the publication "Optimization and repeatability of multi-pool chemical exchange saturation transfer MRI of the prostate at 3.0T" DOI: 10.1002/jmri.26690

They may be useful to anyone interested in implementing Lorentzian linefitting analysis of CEST MRI data.

Primary functions of the code enclosed:

1) Voxelwise and ROI-based fitting of z-spectra using up to 5 Lorentzian lineshapes for water/CEST/NOE pools and a single lineshape (Lorentzian, Gaussian, super-Lorentzian or a straight line) for the MT resonance.
2) Asymmetry calculations from the normalized data.
3) Production of CEST and other parameter maps.
4) Parameter histograms.

The code provided is intended to allow the user to start generating and visualising CEST fitting results quickly. The intention is that this can be used as a basis for more detailed, context-specific analysis. 

A file with phantom test data is provided. This can be used to run the whole script through end-to-end.

If these materials are useful to you and feed into your future work please acknowledge this by referencing the above paper in any resulting publications.

% Vincent Evans
% vincent.evans.14@ucl.ac.uk
% UCL Centre for Medical Imaging
% https://github.com/vsevans/CEST
