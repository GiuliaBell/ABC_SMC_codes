[![DOI](https://zenodo.org/badge/467183547.svg)](https://zenodo.org/badge/latestdoi/467183547)
# ABC-SMC algorithm to parametrise the multi-stage and the exponential models using CFSE data

This repository contains the python codes to carry out the ABC-SMC parameter inference presented as case study in the paper “Multi-stage models of cell proliferation and death: tracking cell divisions with Erlang distributions”. The data files required for the inference are included for both clonotypes.
There are four python files to parametrise the exponential and the multi-stage models making use of OT-I and F5 datasets: ABC_SMC_multi_stage_OTI.py, ABC_SMC_exponential_OTI.py, ABC_SMC_multi_stage_F5.py, ABC_SMC_exponential_F5.py. 
The results of the inference (posterior distributions) are included for each model and clonotype. 
The details of the mathematical models and the inference are presented in the manuscript in Sections 2 and 4 respectively.
