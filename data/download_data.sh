mkdir timebank_data
mkdir -p timebank_data/timebank_dense
mkdir red

#UDS-T
(wget http://decomp.io/projects/time/UDS_T_v1.0.zip && unzip UDS_T_v1.0.zip)

#TemoEval3
(cd timebank_data && wget https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip && unzip tbaq-2013-03.zip)
(cd timebank_data && wget https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/te3-platinumstandard.tar.gz && unzip te3-platinumstandard.tar.gz)

## TimeBank-Dense
(cd timebank_data/timebank_dense && wget https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.T3.txt)
 
