# Bayesian classifier for open-set classification

## Getting Started

<p align="center">
  <img width="800" src="generative model and surrogate class.jpg">
</p>
<p align="justify"> 
We previously developed a hierarchical [Bayesian model](https://arxiv.org/abs/1907.09624) for zero-shot classification of object classes in computer vision (ECCV Workshop 2020). That model established a Bayesian hierarchy among object classes using visual attributes as auxiliary information. To identify both described and undescribed species a similar model is developed by replacing visual attributes with a predefined class hierarchy explicit in the taxonomical classification of biological organisms. More specifically, our proposed method assumes that there are local priors that define the class hierarchy in the feature space (image or DNA) and uses predefined taxonomical classification to build the Bayesian hierarchy around these local priors. 
Our model uses two types of Bayesian priors: global and local. As the name suggests, global priors are shared across all species, whereas local priors are only shared among species belonging to the same genus. Unlike standard Bayesian models where the posterior predictive distribution (PPD) establishes a compromise between prior and likelihood, our approach utilizes posterior predictive distributions to blend local and global priors with data likelihood. Inference for a new insect sample (image or DNA) is performed by evaluating these posterior predictive distributions and assigning the insect to one of the described species that maximizes the posterior predictive likelihood or identifying it as a new species belonging to the surrogate genus class maximizing the posterior predictive likelihood.
 
## Prerequisites

The code was implemented in Matlab 2020. Any version greater 2016 should be fine to run the code.

## Data

You may find the data under `data\INSECTS` folder in this repo. 

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
  
## News
Python version is coming soon, stay tuned!!!

 
