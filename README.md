# PyNeuroSG
a Python toolbox for neural data analysis, developed in Sheinberg lab at Brown University

by Shaobo Guan and Ruobing Xia, created in 2016

This package provides a universual data structure that meets most neurophysiologicla data analysis demands, and provides various functions for bacis and advanced data analsis and visualiztion, including rasters, PSTH, ERP, tuning curve, population decoding, dimensionality reduction, information measurment, variance and correlation analsysis, spectrum analysis, functinal connectivity, current source density (CSD) analsis, recording profile analsis etc

Most functions ares well domumented and some represetitive example usages are demenstrated in `/deme_script`


## Dependencies:
- typical python moduels:
  - numpy
  - scipy
  - matplotlib
  - pandas
  - sklearn
- other modules:
  - dgread (optional for reading stimdg behavioral data, could use pandas instead if behaviral data is in pandas DataFrame)
  - python-neo   
  - [dependencis of neo](http://neo.readthedocs.io/en/latest/install.html)

## Accepted data input:
- Neurophysiology data: any widely used data format in neurophysiology as long as it is supported by [NEO object IO](http://neuralensemble.org/neo/), e.g. TDT, Plexon, BlackRock, and etc.
- Behavioral data: pandas Dataframe or [DLSH dynamic group (dg)](http://charlotte.neuro.brown.edu/private/docs/software/dlsh/dlshdoc.html)

## basis data structure

The core data is a 3D array of shape `[number_of_trials, number_of_timestamps, number_of_channels]` that works for both spiking data and LFP/EEG data, plus a pandas DataFrame that stored task/behaviorial information related to every trial

## Getting started guide
  
  Gettting started following the Jupyter notebook `demo_script/Demo_Data_Structure.ipynb`
  
  More demos can be found under `demo_script`.
  
  Usage of specific functions can be found in the doc strirng, e.g. `help(pnp.PsthPlot)`
