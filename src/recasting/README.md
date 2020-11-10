# Creation of NLI Dataset

This directory contains the scripts to create the five different NLI recast datasets obtained from the four original datasets -- UDST, TempEval3, TimeBank-Dense, and RED.  

Before running the NLI recasting script, you need to do the following:
1. Download the original data in the `data/` directory of this repo (details [here](https://github.com/sidsvash26/temporal_nli/tree/main/data)).

Once you have downloaded all the original data, you can simply run the following from the terminal to recast all datasets 
```shell
bash run_all_recast.sh 
```
(Note that the underlying scripts use Python dependencies which can be downloaded from the requirements file)

The bash script will create train, dev and test splits for each recasted dataset in the `/data` folder of this repo


