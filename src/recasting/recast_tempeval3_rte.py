import pandas as pd
import numpy as np
from lemminflect import getInflection
from data_loader_timebank import load_obj
import argparse
import json
import random
from recast_utils import *
SEED = 2
'''

Usage: python recast_tempeval3_rte.py \
       --inputdata "tempeval-3-all.csv" \
       --out_train "../../data/train/"  \
       --out_dev "../../data/dev/"  \
       --out_test "../../data/test/" 

'''
def create_tb_split(docid, dev_docs, test_docs):
    '''
    docid in Tempeval 3

    dev_docs: a list of doc names in dev
    test_docs: a list of doc names in test
    '''
    if docid in dev_docs:
        return "dev"
    elif docid in test_docs:
        return "test"
    else:
        return "train"
    
def reltype_to_finegrained_prototype(rel):
    '''
    for a given reltype, construct prototype values
    '''
    ####################
    if rel=="BEFORE":
        return [1,10,
                30,60]
    elif rel=="AFTER":
        return [30,60,
                1,10]
    ####################
    elif rel=="IBEFORE":
        return [1,30,
               30,60]
    elif rel=="IAFTER":
        return [30,60,
                1,30]
    ####################
    elif rel=="BEGINS":
        return [1,30,
                1,70]
    elif rel=="BEGUN_BY":
        return [1,70,
                1,30]
    ####################
    elif rel=="ENDS":
        return [70, 90, 
                10, 90]
    elif rel=="ENDED_BY":
        return [10, 90, 
                70, 90]
    ####################
    elif rel=="IS_INCLUDED":
        return [30, 60,
               10, 80]
    elif rel=="INCLUDES":
        return [10, 80,
               30, 60]
    #####################
    elif rel=="DURING":
        return [1,40,
               20,70]
    elif rel=="DURING_INV":
        return [20,70,
               1,40]
    #####################
    elif rel=='SIMULTANEOUS' or rel=='IDENTITY':
        return [20,60,
               20,60]
    else:
        raise ValueError('No Value met by relType')
        
        
def relation_vector(finegrained_input):
    '''
    Create a Relation vector from UDS-Time sliders
    
    vector: 8-dimensional vector (8 dimensions are commented below)
    '''

    b1, e1, b2, e2 = finegrained_input
    
    ans = [0]*8
    
    ## X started before Y started
    if b1 < b2:
        ans[0] = 1
    ## X started before Y ended
    if b1 < e2:
        ans[1] = 1
    ## X ended before Y started
    if e1 < b2:
        ans[2] = 1
    ## X ended before Y ended
    if e1 < e2:
        ans[3]=1
        
    ## Y started before X started
    if b2 < b1:
        ans[4] = 1
    ## Y started before X ended
    if b2 < e1:
        ans[5] = 1
    ## Y ended before X started
    if e2 < b1:
        ans[6] = 1
    ## Y ended before X ended
    if e2 < e1:
        ans[7]=1
        
    return ans

def create_relation_NLI(row,
                        pairid = 0,
                        skipcount=0,
                       extra_info=False):
    '''
    Given an event-pair id create NLI pairs
    data: UDS Time dataframe with some extra features
    '''
    
    recasted_data = []
    recasted_metadata = []
    
    delim = "|"
    full_event_pair_id = row['corpus']+delim+row['docid']+delim+row['eventInstanceID']+delim+ row['relatedToEventInstance']

    combined_sent_lemmas = combined_sentence_lemma(row['eid1_sent_conllu'],
                                                   row['eid2_sent_conllu'],
                                                   row['eid1_sent_id'],
                                                   row['eid2_sent_id'])

    premise = row['combined_sent']

    split = row['split']
        
    tokenid1 = row['combined_tokenid1']
    tokenid2 =  row['combined_tokenid2']
    len_tokenid1 = len(tokenid1.split("_"))
    len_tokenid2 = len(tokenid2.split("_"))
    
    ## Skip if a predicate is neither a verb nor has a copula:
    if (row['eid1_POS']!='VERB' and len(row['eid1_text_full'].split())<2) or (row['eid2_POS']!='VERB' and len(row['eid2_text_full'].split())<2):
        skipcount+=1
        return [], [], pairid, skipcount

    ## Skip if multiple tokenids (since POS of root is not recognizable)
    if len_tokenid1>1 or len_tokenid2>1:
        skipcount+=1
        return [], [], pairid, skipcount

    ##### Predicate 1 ######
    pred1_text = row['eid1_text']
    pred1_pos = row['eid1_POS']
    pred1_lemma = row['eid1_stanza_lemma']

    # Check if predicate ambiguity exists, add DOBJ boolean 
    if combined_sent_lemmas.count(pred1_lemma) > 1:
        dobj_bool_pred1 = True
    else:
        dobj_bool_pred1 = False

    dependency_nodes_pred1 = DependencyGraph(row['eid1_sent_conllu'],
                                            top_relation_label='root').nodes
    sentence_tokens_pred1 = row['eid1_sentence'].split()

    full_event_text1 = create_full_event_text_tb(row['eid1_token_id'],
                                                  row['eid1_text_full'],
                                                  row['eid1_POS'],
                                                  pred1_lemma,
                                                  dependency_nodes_pred1,
                                                  sentence_tokens_pred1,
                                                  dobj_bool=dobj_bool_pred1)
    ##### Predicate 2 ######
    pred2_text = row['eid2_text']
    pred2_pos = row['eid2_POS']
    pred2_lemma = row['eid2_stanza_lemma']

    # Check if predicate ambiguity exists, add DOBJ boolean 
    if combined_sent_lemmas.count(pred2_lemma) > 1:
        dobj_bool_pred2 = True
    else:
        dobj_bool_pred2 = False
    
    dependency_nodes_pred2 = DependencyGraph(row['eid2_sent_conllu'],
                                            top_relation_label='root').nodes
    sentence_tokens_pred2 = row['eid2_sentence'].split()

    full_event_text2 = create_full_event_text_tb(row['eid2_token_id'],
                                                  row['eid2_text_full'],
                                                  row['eid2_POS'],
                                                  pred2_lemma,
                                                  dependency_nodes_pred2,
                                                  sentence_tokens_pred2,
                                                  dobj_bool=dobj_bool_pred2)

    entailment_threshold = 1
    relation_votes = row['relation_vector']
    
    hypothesis_list = relation_hypothesis(full_event_text1,
                                            full_event_text2,
                                            pred1_pos,
                                            pred2_pos)
                                    
    for hypothesis, relation_vote in zip(hypothesis_list, relation_votes):
        pairid +=1
        temp_dict = {}
        temp_dict['context'] = premise        
        temp_dict['hypothesis'] = string_expansions(hypothesis)
        
        if relation_vote >=entailment_threshold:
            temp_dict['label'] = 'entailed'
        else:
            temp_dict['label'] = 'not-entailed'
            
        temp_dict['pair-id'] = pairid
        temp_dict['split'] = split
        temp_dict['type-of-inference'] = 'temporal-relation'

        if extra_info:
            temp_dict['pred1_text_raw'] = pred1_text
            temp_dict['pred1_lemma'] = pred1_lemma
            temp_dict['pred1_upos'] = pred1_pos
            temp_dict['pred2_text_raw'] = pred2_text
            temp_dict['pred2_lemma'] = pred2_lemma
            temp_dict['pred2_upos'] = pred2_pos
            temp_dict['corpus-sent-id'] = full_event_pair_id

        ##Metadata
        metadata_dict = {}
        metadata_dict['corpus'] = 'tempeval-3'
        metadata_dict['corpus-license'] = 'todo'
        metadata_dict['corpus-sent-id'] = full_event_pair_id
        metadata_dict['creation-approach'] = 'automatic'
        metadata_dict['pair-id'] = pairid

        recasted_data.append(temp_dict)
        recasted_metadata.append(metadata_dict)
        
    return recasted_data, recasted_metadata, pairid, skipcount

def main():
        # Data Locations
    parser = argparse.ArgumentParser(
        description='Recast Tempeval3 to NLI format.')
    parser.add_argument('--inputdata', type=str,
                        default='tempeval-3-all.csv',
                        help='tempeval-3-all csv output from data_loader_timebank.')

    parser.add_argument('--out_train', type=str,
                        default='train/',
                        help='recasted train data folder location ')

    parser.add_argument('--out_dev', type=str,
                        default='dev/',
                        help='recasted dev data folder location ')

    parser.add_argument('--out_test', type=str,
                        default='test/',
                        help='recasted test data folder location ')

    args = parser.parse_args()


    # ### Import dataframe
    te3 = pd.read_csv(args.inputdata)
    
    ## Add columns to dataframe:
    te3['eid1_stanza_POS'] = te3.apply(lambda row: extract_stanza_info(row, 1, param="ctag"), axis=1)
    te3['eid2_stanza_POS'] = te3.apply(lambda row: extract_stanza_info(row, 2, param="ctag"), axis=1)
    te3['eid1_stanza_lemma'] = te3.apply(lambda row: extract_stanza_info(row, 1, param="lemma"), axis=1)
    te3['eid2_stanza_lemma'] = te3.apply(lambda row: extract_stanza_info(row, 2, param="lemma"), axis=1)
    ## Add missing GOLD POS
    ## Replace
    te3['eid1_POS'] = te3['eid1_POS'].replace(['ADJECTIVE','UNKNOWN','PREP', 'PREPOSITION'],
                                              ['ADJ','OTHER','OTHER','OTHER'])
    te3['eid2_POS'] = te3['eid2_POS'].replace(['ADJECTIVE','UNKNOWN','PREP', 'PREPOSITION'],
                                              ['ADJ','OTHER','OTHER','OTHER'])
    te3['eid1_POS'] = te3.apply(lambda row: fill_missing_POS(row, 1), axis=1 )
    te3['eid2_POS'] = te3.apply(lambda row: fill_missing_POS(row, 2), axis=1 )

    te3['eid1_text_full'] = te3.apply(lambda row: extract_predpatt_text(row, 1), axis=1)
    te3['eid2_text_full'] = te3.apply(lambda row: extract_predpatt_text(row, 2), axis=1)

    #### Add relation vector feature to data:
    te3['relation_vector'] = te3.apply(lambda row: relation_vector(reltype_to_finegrained_prototype(row.relType)), axis=1)


    ## Create a list of dev docs by randomly sampling documents from train data
    #train_full_docs = list(te3[te3.corpus!='te3-platinum']['docid'].unique())
    test_docs = te3[te3.corpus=='te3-platinum']['docid'].unique() ## entire te3-platinum corpus is test set
    #dev_docs = random.Random(SEED).sample(train_full_docs, len(test_docs))
    
    #dev_docs are generated from the command above and listed here for reproducibility
    dev_docs = ['PRI19980205.2000.1890', 'wsj_0520', 'wsj_0806', 'wsj_0505', 'wsj_0674', 
                'wsj_0938', 'wsj_1014', 'wsj_0176', 'wsj_0168', 'wsj_0144', 'wsj_0184', 
                'ea980120.1830.0071', 'APW19980807.0261', 'NYT19981026.0446', 'APW19980818.0515', 
                'NYT20000105.0325', 'NYT20000406.0002', 'APW20000405.0276', 'NYT20000330.0406']
    
    ## Create a split column in the data
    te3['split'] = te3.docid.map(lambda x: create_tb_split(x, dev_docs, test_docs))

    train = te3[te3['split']=='train']
    dev = te3[te3['split']=='dev']
    test = te3[te3['split']=='test']

    #######################################################
    ## Recast Data
    #######################################################
    
    pairid = -1  # count total pair ids
    # Count event-pairs skipped due to ambiguous text for highlighting predicate.
    skipcount = 0

    #### RECAST TRAIN #########
    data = []
    metadata = []

    for idx, row in train.iterrows():

        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                    pairid = pairid,
                                                                                    skipcount=skipcount,
                                                                                   extra_info=False)
        #print(f"Row ids: {idx}")
        if recasted_data:
                data += recasted_data
                metadata += recasted_metadata

        if (pairid+1)%(500)==0:
            print(f"Total pair-ids processed so far: {pairid+1}")
    
    with open(args.out_train + "recast_tempeval3_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_train + "recast_tempeval3_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)\

    #### RECAST DEV #########
    data = []
    metadata = []

    for idx, row in dev.iterrows():

        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                    pairid = pairid,
                                                                                    skipcount=skipcount,
                                                                                   extra_info=False)


        if recasted_data:
                data += recasted_data
                metadata += recasted_metadata


        if (pairid+1)%(500)==0:
            print(f"Total pair-ids processed so far: {pairid+1}")
    

    with open(args.out_dev + "recast_tempeval3_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_dev + "recast_tempeval3_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)

    #### RECAST TEST #########
    data = []
    metadata = []

    for idx, row in test.iterrows():
        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                    pairid = pairid,
                                                                                    skipcount=skipcount,
                                                                                   extra_info=False)


        if recasted_data:
                data += recasted_data
                metadata += recasted_metadata


        if (pairid+1)%(500)==0:
            print(f"Total pair-ids processed so far: {pairid+1}")
    
    with open(args.out_test + "recast_tempeval3_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_test + "recast_tempeval3_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)


    print(f"Total pair-ids: {pairid+1}")
    print(f'Total events skipped: {skipcount}')

if __name__== "__main__":
	main()
