# Bayesian classifier for open-set classification

## Getting Started

This is a ReadMe file for running the Bayesian classifier and reproduce the results presented in the paper. 

## Prerequisites

The code was implemented in Matlab 2020. Any version greater 2016 should be fine to run the code.

## Data

You may download the data from this anonymous [link](). Please put dataset into `data` folder and move the `data` folder into the same directory which contains the folders for codes.

Note that there are 2 data files: `data.mat` and 	`splits.mat`. The  variables and their explanations are listed below:

`data.mat`
* `embeddings_dna`: Embeddings for DNA data
* `embeddings_img`: Embeddings for IMAGE data
* `labels`: Numeric labels for species
* `species`: Species names 
* `G`: Genus labels of species
* `nucleotides`: DNA barcode of the species
* `bold_ids`: IDs of the sampels from BOLD system. You may use this ids to see the full details of the spekciemen in BOLD system.
* `ids`: Image names of the samples in our dataset.

`splits.mat`
* `train_loc`: Indices of training data points for tuning
* `trainval_loc`: Indices of training data points for final inference
* `test_seen_loc`: Indices of test data from seen classes
* `test_unseen_loc`: Indices of test data from unseen classes
* `val_seen_loc`: Indices of validation data from seen classes
* `val_unseen_loc`: Indices of validation data from unseen classes



## Experiments

To reproduce the results from the paper, open the `Demo.m` script and specify the dataset and model version. Please change the datapath to your project path in `Demo.m` script.

If you want to perform hyperparameter tuning, please set the `tuning=true` in  `Demo.m` script.

 
