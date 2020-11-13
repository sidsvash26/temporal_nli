# Train from scratch or use (evaluate) Roberta models

This directory contains the scripts to create the run training on the recasted datasets from scratch. 
There are three main scripts that train different NLI recasted datasets:

1. `train_uds_duration.sh`: This script will train the recasted UDS\_duration dataset. 
2. `train_uds_order.sh`: This script will train the recasted UDS\_order dataset.
3. `train_order_others.sh`: This script will train TBD, TE3, and RED recasted datasets. 

We mainly use huggingface's transformers library to train these models but we use some of the scripts in this directory (modified from huggingace library) to save specific checkpoints info.

## Evaluation 
Evaluation of our best models can be run from the notebook: `Evaluate_models.ipynb`

## Data Statistics
Once you have run the recasting scripts, you can check data statistics by running the `Data_stats.ipynb` notebook.

The training and dev sets of recasted datasets are stored in  `data/train` and `data\dev` of this repo once you have run the recasting scripts after following the instructions in `src/recasting`. 

