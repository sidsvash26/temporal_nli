# Datasets

This directory should contain the original directories of each of the 4 datasets described below. Once the recast scripts are run, the NLI dataset will be generated in the current directory under `train/` `dev/` and `test/` sub-directories.

The URLs of each dataset is listed below:
1. [UDS-T Duration and Order](http://decomp.io/projects/time/)
2. [TempEval-3](https://www.cs.york.ac.uk/semeval-2013/task1/index.php%3Fid=data.html)
3. [TimeBank-Dense](https://www.usna.edu/Users/cs/nchamber/caevo/)
4. [Richer Event Description](https://catalog.ldc.upenn.edu/LDC2016T23)

You can download datasets 1, 2 and 3 by running the following script in bash:
```bash
bash download_data.sh
``` 
RED corpora has no publicly available webpage so you need to get the data from LDC and then copy it a subdirectory named `data/red/`
