# Longitudinal prediction of disability progression

Code to reproduce the experiments from the paper : "Longitudinal machine learning modeling of MS patient trajectories improves predictions of disability progression"

## Installation

`cd ms`

`pip install -e . `

Extra dependencies required from training the models : 

- Macau (https://github.com/jaak-s/macau)

- Optunity (https://optunity.readthedocs.io)


## Cleaning and pre-processing of the MSBase Data

This pre-processing script has been developped based on a data cut from MSBase from 2018.

To build the clean dataset, run 

`ms/utils/data_preprocessing.py`

The data files are expected to be found at the location : `~/Data/MS/Cleaned_MSBASE/` but this location can be changed directly in the scripts.

To create the folds indices (patient indices for train/val/test in all 5 folds) : 

`python ms/utils/generate_folds`

## Static Model

To train the static model  :

` cd ms/Setup1/`

`python static_model.py`

The models and results will be saved in `ms/comparisons_results/Setup1`


## Dynamic Model

To train the dynamic model  :

` cd ms/Setup1/`

`python dynamic_model.py`

The models and results will be saved in `ms/comparisons_results/Setup1`


## BPTF Model

To train the bptf model  :

` cd ms/Setup1/`

`python model_6.py`

The models and results will be saved in `ms/comparisons_results/`




