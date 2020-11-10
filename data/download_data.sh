mkdir raw_data

mkdir -p raw_data/timebank_data
mkdir -p raw_data/timebank_data/timebank-dense
mkdir -p raw_data/red

#UDS-T
(cd raw_data && wget http://decomp.io/projects/time/UDS_T_v1.0.zip && unzip UDS_T_v1.0.zip)
## UD EWT raw data
(cd raw_data && wget https://github.com/UniversalDependencies/UD_English-EWT/archive/r1.3.zip && unzip r1.3.zip)

#TemoEval3
(cd raw_data/timebank_data && wget https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip && unzip tbaq-2013-03.zip)
(cd raw_data/timebank_data && wget https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/te3-platinumstandard.tar.gz && tar -xzf te3-platinumstandard.tar.gz)

## TimeBank-Dense
(cd raw_data/timebank_data/timebank-dense && wget https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.T3.txt)
 
