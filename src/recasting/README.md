# Creation of NLI Dataset

This directory contains the scripts to create the five different NLI recast datasets obtained from the following four original datasets:
1. [UDS-T Duration and Order](http://decomp.io/projects/time/UDS_T_v1.0.zip)
2. [TempEval-3](https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip)
3. [TimeBank-Dense](https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.T3.txt)
4. Richer Event Description (LDC Catalog No.: LDC2016T23)

You need to copy these original datasets into the `data/` folder of this repository for the scripts to run.
You also need to copy the [Unimorph english](https://github.com/unimorph/eng/blob/master/eng) to the current directory.

Once you have copied the respective files, you can simply run the following from the terminal 
```shell
bash run_all_recast.sh 
```
(Note that the underlying scripts use Python dependencies which can be downloaded from the requirements file)

The bash script will create train, dev and test splits for each recasted dataset in the `/data` folder of this repo


