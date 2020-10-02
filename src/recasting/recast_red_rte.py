'''
Author: Siddharth Vashishtha
Affiliation: University of Rochester
Creation Date: 28th May, 2020

Process RED Dataframe and recast into RTE format

Usage:

Usage: python recast_red_rte.py \
   --inputdata "red-all.csv" \
   --out_train "../../data/train/"  \
   --out_dev "../../data/dev/"  \
   --out_test "../../data/test/"

'''

import pandas as pd
import numpy as np
from lemminflect import getInflection
from data_loader_timebank import load_obj
import argparse
import json
import re
from recast_utils import *
from bs4 import BeautifulSoup


def remove_xml_tags(s):
    s = re.sub(r'<[^>]+>', r' ', s)
    ##convert double space to single
    s = re.sub(r'  ', r' ', s)

    return s


def reltype_to_finegrained_prototype(rel):
    '''
    for a given reltype in RED-corpus, construct prototype values
    '''
    ####################
    if "BEFORE" in rel:  # 'before' in te3
        return [1, 10,
                30, 60]
    ####################
    elif rel == "BEGINS-ON":  #'begins' in te3
        return [1,30,
                1,70]
    #####################
    elif rel == "ENDS-ON":  # 'ends' in te3
        return [70, 90, 
                10, 90]
    #####################
    elif "CONTAINS" in rel:  # 'includes' in te3
        return [10, 80,
                30, 60]
    #####################
    elif rel == 'SIMULTANEOUS':  # simultaneous
        return [20, 60,
                20, 60]
    ####################
    elif 'OVERLAP' in rel:
        # dummy values overwritten in ambiguous layer
        return [0,0,0,0]

    else:
        raise ValueError('No Value met by red-relation')

def relation_vector(finegrained_input):
    '''
    Create a Relation vector from UDS-Time sliders

    vector: 8-dimensional vector (8 dimensions are commented below)
    '''

    b1, e1, b2, e2 = finegrained_input

    ans = [0]*8

    # X started before Y started
    if b1 < b2:
        ans[0] = 1
    # X started before Y ended
    if b1 < e2:
        ans[1] = 1
    # X ended before Y started
    if e1 < b2:
        ans[2] = 1
    # X ended before Y ended
    if e1 < e2:
        ans[3] = 1

    # Y started before X started
    if b2 < b1:
        ans[4] = 1
    # Y started before X ended
    if b2 < e1:
        ans[5] = 1
    # Y ended before X started
    if e2 < b1:
        ans[6] = 1
    # Y ended before X ended
    if e2 < e1:
        ans[7] = 1

    return ans

def ambiguous_layer(rel, relation_vector):
    '''
    Given a relation and its corresponding relation vector
    edit some values in the relation vector to -1 denoting that
    those cases are neither "entailed" nor "not-entailed"
    '''
    ans = list(relation_vector)

    if rel=="BEFORE":
        # X ended before Y started
        ans[2] = -1

        # Y started before X ended
        ans[5] = -1

        return ans

    elif rel=="BEGINS-ON":
        # X ended before Y ended
        ans[3] = -1

        # Y ended before X ended
        ans[7] = -1

        return ans

    elif rel=="ENDS-ON":
        # X started before Y started
        ans[0] = -1

        # Y started before X started
        ans[4] = -1

        return ans

    elif 'OVERLAP' in rel:
        ## overlap is ambiguous for all templates
        ans = [-1]*8

    else:
        return ans


def create_relation_NLI(row,
                        pairid=0,
                        skipcount=0,
                        extra_info=False):
    '''
    Given an event-pair id create NLI pairs
    data: UDS Time dataframe with some extra features
    '''
    recasted_data = []
    recasted_metadata = []

    ## skip overlap relations as they are ambiguous wrt to allen relations
    if 'OVERLAP' in row['relation']:
        skipcount += 1
        return [], [], pairid, skipcount

    delim = "|"
    full_event_pair_id = row['corpus']+delim+row['docid']+delim + \
        row['eventInstanceID']+delim + row['relatedToEventInstance']
    combined_sent_lemmas = combined_sentence_lemma(row['eid1_sent_conllu'],
                                                   row['eid2_sent_conllu'],
                                                   row['eid1_sent_id'],
                                                   row['eid2_sent_id'])

    premise = row['combined_sent']
    split = row['split']

    tokenid1 = row['combined_tokenid1']
    tokenid2 = row['combined_tokenid2']
    len_tokenid1 = len(str(tokenid1).split("_"))
    len_tokenid2 = len(str(tokenid2).split("_"))

    # Skip for multiple spanning predicates
    if len_tokenid1 > 1 or len_tokenid2 > 1:
        skipcount += 1
        return [], [], pairid, skipcount

    ## Skip if a predicate is neither a verb nor has a copula:
    if (row['eid1_POS']!='VERB' and len(row['eid1_text_full'].split())<2) or (row['eid2_POS']!='VERB' and len(row['eid2_text_full'].split())<2):
        skipcount+=1
        return [], [], pairid, skipcount

    ##### Predicate 1 ######
    pred1_text = row['eid1_text']
    pred1_pos = row['eid1_POS']
    pred1_lemma = row['eid1_lemma']

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
    pred2_lemma = row['eid2_lemma']

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

    
    relation_vectors = ambiguous_layer(row['relation'], row['relation_vector'])

    hypothesis_list = relation_hypothesis(full_event_text1,
                                            full_event_text2,
                                            pred1_pos,
                                            pred2_pos)

    for hypothesis, relation_vector in zip(hypothesis_list, relation_vectors):
        temp_dict = {}
        temp_dict['context'] = str(BeautifulSoup(remove_xml_tags(premise), features='lxml').text)
        temp_dict['hypothesis'] = string_expansions(hypothesis)

        if relation_vector == 1:
            temp_dict['label'] = 'entailed'
        elif relation_vector == 0:
            temp_dict['label'] = 'not-entailed'
        else:
            ## skip ambiguous templates
            continue

        pairid += 1

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

        # Metadata
        metadata_dict = {}
        metadata_dict['corpus'] = 'RED'
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
        description='Recast RED to NLI format.')
    parser.add_argument('--inputdata', type=str,
                        default='red-all.csv',
                        help='red full data output from data_loader_red.')

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

    te3['eid1_text_full'] = te3.apply(lambda row: extract_predpatt_text(row, 1), axis=1)
    te3['eid2_text_full'] = te3.apply(lambda row: extract_predpatt_text(row, 2), axis=1)
    # Add relation vector feature to data:
    te3['relation_vector'] = te3.apply(lambda row: relation_vector(
                                        reltype_to_finegrained_prototype(row.relation)), axis=1)

    train = te3[te3['split'] == 'train']
    dev = te3[te3['split'] == 'dev']
    test = te3[te3['split'] == 'test']

    #######################################################
    # Recast Data
    #######################################################

    pairid = -1  # count total pair ids
    # Count event-pairs skipped due to ambiguous text for highlighting predicate.
    skipcount = 0

    #### RECAST TRAIN #########
    data = []
    metadata = []

    for idx, row in train.iterrows():
        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                  pairid=pairid,
                                                                                  skipcount=skipcount,
                                                                                  extra_info=False)

        if recasted_data:
            data += recasted_data
            metadata += recasted_metadata

        if (pairid+1) % (500) == 0:
            print(f"Total pair-ids processed so far: {pairid+1}")

    with open(args.out_train + "recast_red_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_train + "recast_red_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)\

    #### RECAST DEV #########
    data = []
    metadata = []

    for idx, row in dev.iterrows():

        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                  pairid=pairid,
                                                                                  skipcount=skipcount,
                                                                                  extra_info=False)

        if recasted_data:
            data += recasted_data
            metadata += recasted_metadata

        if (pairid+1) % (500) == 0:
            print(f"Total pair-ids processed so far: {pairid+1}")

    with open(args.out_dev + "recast_red_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_dev + "recast_red_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)

    #### RECAST TEST #########
    data = []
    metadata = []

    for idx, row in test.iterrows():
        recasted_data, recasted_metadata, pairid, skipcount = create_relation_NLI(row,
                                                                                  pairid=pairid,
                                                                                  skipcount=skipcount,
                                                                                  extra_info=False)

        if recasted_data:
            data += recasted_data
            metadata += recasted_metadata

        if (pairid+1) % (500) == 0:
            print(f"Total pair-ids processed so far: {pairid+1}")

    with open(args.out_test + "recast_red_data.json", 'w') as out_data:
        json.dump(data, out_data, indent=4)
    with open(args.out_test + "recast_red_metadata.json", 'w') as out_data:
        json.dump(metadata, out_data, indent=4)

    print(f"Total pair-ids: {pairid+1}")
    print(f'Total events skipped: {skipcount}')


if __name__ == "__main__":
    main()

