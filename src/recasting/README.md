# Creation of NLI Dataset

This directory contains the scripts to create the five different NLI recast datasets obtained from the four original datasets -- UDST, TempEval3, TimeBank-Dense, and RED. The details on downloading each original dataset are [here](https://github.com/sidsvash26/temporal_nli/tree/main/data). 

You need to copy these original datasets into the `data/` folder of this repository for the scripts to run.
You also need to copy the [Unimorph english](https://github.com/unimorph/eng/blob/master/eng) to the current directory.

Once you have copied the respective files, you can simply run the following from the terminal 
```shell
bash run_all_recast.sh 
```
(Note that the underlying scripts use Python dependencies which can be downloaded from the requirements file)

The bash script will create train, dev and test splits for each recasted dataset in the `/data` folder of this repo


