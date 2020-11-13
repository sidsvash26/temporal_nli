###
# This scripts assumes that the raw data is present in "../../data" folder
# This script recasts the following 5 datasets into an NLI format:
# 1. UDST_Durarion
# 2. UDST_Relation
# 3. TempEval3
# 4. TimeBank-Dense
# 5. RED Corpus
#
###

#Download and create Unimorph Dictionariesi
echo "Downloading English unimorph file"
if [ ! -e "eng" ]; then
       wget https://raw.githubusercontent.com/unimorph/eng/master/eng
else
        echo "eng file already present in the current dir"
fi

echo "Creating Unimorph dictionaries"
if [ ! -e "lemma_to_inflection_dict.pkl" ] && [ ! -e "word_to_lemma_dict.pkl" ]; then
       python unimorph_dicts.py
else
        echo "Unimorph dicts already present in the current dir"
fi

## Create train. dev. test dirs
mkdir -p "../../data/train/"
mkdir -p "../../data/dev/"
mkdir -p "../../data/test/"

## Recast UDS-T Duration
echo "Initialising recasting of UDST Duration"
python recast_temporal_duration_rte.py \
                --udstime "../../data/raw_data/UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv" \
                --out_train "../../data/train/" \
                --out_dev "../../data/dev/" \
                --out_test "../../data/test/"

## Recast UDS-T Relation
echo "Initialising recasting of UDST Relation"
python recast_temporal_relation_rte.py \
            --udstime "../../data/raw_data/UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv" \
            --out_train "../../data/train/" \
            --out_dev "../../data/dev/" \
            --out_test "../../data/test/"

## Recast TempEval3
#Load data into pandas dataframe
echo "Saving TempEval3 data in a pandas dataframe"
python data_loader_timebank.py

#Recast
echo "Recasting TempEval3 data"
python recast_tempeval3_rte.py \
       --inputdata "tempeval-3-all.csv" \
       --out_train "../../data/train/"  \
       --out_dev "../../data/dev/"  \
       --out_test "../../data/test/"

## Recast TimeBank-Dense
#Load data into pandas dataframe
echo "Saving TBD data in a pandas dataframe"
python data_loader_timebank_dense.py

#Recast
echo "Recasting TB-Dense data"
python recast_tbdense_rte.py \
   --inputdata "timebank-dense-all.csv" \
   --out_train "../../data/train/"  \
   --out_dev "../../data/dev/"  \
   --out_test "../../data/test/"

## Recast RED corpus
#Load data into pandas dataframe
echo "Saving RED data in a pandas dataframe"
python data_loader_red.py

#Recast
echo "Recasting RED data"
python recast_red_rte.py \
   --inputdata "red-all.csv" \
   --out_train "../../data/train/"  \
   --out_dev "../../data/dev/"  \
   --out_test "../../data/test/"

#Convert json data files to tsvs
python create_recasted_tsv.py
