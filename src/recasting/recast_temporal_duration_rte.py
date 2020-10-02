#!/usr/bin/env python
# coding: utf-8

'''
Dependencies:
 - PredPatt: https://github.com/hltcoe/PredPatt
 - doc_utils: python file inlcuded in this folder
 - pandas
 - numpy
 - UD_EWT v1.3 data files (https://github.com/UniversalDependencies/UD_English-EWT/releases/tag/r1.3)
 - nltk

Usage: python recast_temporal_duration_rte.py \
                --udstime "../../data/UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv" \
                --out_train "../../data/train/" \
                --out_dev "../../data/dev/" \
                --out_test "../../data/test/" 
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import doc_utils
from collections import defaultdict, Counter
from itertools import product
from tqdm import tqdm
#import matplotlib.pyplot as plt
import re
import json
import argparse
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
from nltk import DependencyGraph
import re
from recast_utils import *

options = PredPattOpts(resolve_relcl=True, 
                        borrow_arg_for_relcl=True, 
                        resolve_conj=False, 
                        cut=True)

#Data Locations:
ud_path = "../../data/UD_English-EWT-r1.3/"
ud_train  =  "../../data/UD_English-EWT-r1.3/en-ud-train.conllu"
ud_dev  =  "../../data/UD_English-EWT-r1.3/en-ud-dev.conllu"
ud_test  =  "../../data/UD_English-EWT-r1.3/en-ud-test.conllu"
ud_data = [ud_train, ud_dev, ud_test]

# #### Hypothesis Generation Functions
duration_dict = defaultdict(str)
duration_dict[0] = 'instantaneously'
duration_dict[1] = 'second'
duration_dict[2] = 'minute'
duration_dict[3] = 'hour'
duration_dict[4] = 'day'
duration_dict[5] = 'week'
duration_dict[6] = 'month'
duration_dict[7] = 'year'
duration_dict[8] = 'decade'
duration_dict[9] = 'century'
duration_dict[10] = 'eternity'


def get_structs(ud_path):
    files = ['en-ud-train.conllu', 'en-ud-dev.conllu', 'en-ud-test.conllu']
    structures = {}
    for file in files:
        with open(ud_path + file, 'r') as f:
            iden = 0
            a = ""
            words = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                else:
                    iden += 1
                    a = a
                    structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = sent
                    sent_id = file + " " + str(iden)
                    structures[sent_id] = structure
                    a = ""
                    words = []
    return structures

structures = get_structs(ud_path)
print(f"NLTK Dependency strucutures stored in memory")

def get_predicate_pos(row, ewt, event=1):
    '''
    Input row: row of UDS-Time dataframe
    
    Event: 1 or 2 (predicate 1 or 2)
    '''
    event_tokenid = getattr(row,"Pred" + str(event) + ".Mod.Token")
    sentid1, sentid2 =getattr(row,"Combined.Sent.ID").split("|")
     
    combined_pos = ewt.uds_syntax(doc_utils.conllu_to_decomp(sentid1), 
                                  item="upos") +ewt.uds_syntax(doc_utils.conllu_to_decomp(sentid2), 
                                                               item="upos")
                    
    return combined_pos[event_tokenid]

def article(duration_value):
    '''articles for corresponding duration label integer'''
    if duration_value==10: ## eternity
        return " "
    elif duration_value==3: ## hour
        return " an "
    else:
        return " a "
    
def modal(duration_value):
    '''
    modal string based on duration value
    '''
    ## Keeping constant for all duration values, can be changed later
    return "did take or will take"
    
def modify_string(s):
    s = re.sub("did take or will take shorter than a second.", "did happen or will happen instantaneously.", str(s))
    s = re.sub("did take or will take longer than eternity.", "is always true.", str(s))
    s = re.sub("did take or will take shorter than eternity.", "is not always true.", str(s))
    # s = re.sub("but shorter than eternity.", "but is not always true.", str(s))
    s = re.sub(" n't ", " not ", str(s))
    s = re.sub(" nt ", " not ", str(s))
    return s

def create_full_event_text(event_id,
                          event_text,
                          event_pos,
                          event_lemma,
                          ewt,
                          dobj_bool=False):
    '''
    Output: create event_full_text based on dependencies, pos, and inflections
    '''
    sentid, tokenid = event_id.split("_")
    dependency_nodes = structures[sentid].nodes
    dependency_tokenids = find_dependency_tokenids(dependency_nodes, int(tokenid),
                                                    dobj_bool=dobj_bool) 
    sentence_tokens = ewt.decomp_graph[doc_utils.conllu_to_decomp(sentid)].sentence.split()
    
    full_event_text = []
    #print(f"dep tokenids: {dependency_tokenids}")
    #print(f"event text: {event_text}")
    
    ## Inflection (Only changes for verbs):
    #print(f"Sentence: {' '.join(sentence_tokens)}")
    event_text_inflected = unimorph_inflect(event_lemma, 
                                            event_pos, 
                                            event_text)

    ## Add nearby dependency of the main predicate
    for dep_tokenid in dependency_tokenids:
        if dep_tokenid != int(tokenid):
            #print(event_id)
            full_event_text.append(sentence_tokens[dep_tokenid])
        else:
            full_event_text.append(event_text_inflected)
            
    ## In case of copula, return the text after copula
    if len(event_text.split()) > 1:
        return " ".join(event_text.split()[1:])
    
    else:
        return " ".join(full_event_text)

def duration_hypothesis_entailed(event_id, 
                                 event_text, 
                                 event_pos,
                                 event_lemma,
                                 correct_duration, 
                                 ewt,
                                 window=1,
                                 dobj_bool=False):
    '''
    Based on the sliding window (default = 1), generate entailed
    hypothesis
    
    correct_duration = List of duration labels for the given event
                        (list has only 1 item for train, 3 for devtest)
                        
                        
                        
                        event_id, 
                                                       pred_text_original, 
                                                       pred_upos,
                                                       pred_lemma,
                                                       pred_duration, 
                                                       window=sliding_window
    
    '''
    
    ## Create full event_text based on dependencies of the predicate
    full_event_text = create_full_event_text(event_id, 
                                             event_text, 
                                             event_pos, 
                                             event_lemma,
                                             ewt,
                                             dobj_bool=dobj_bool)
        
    ## Duration calculation
    min_duration = min(correct_duration)
    max_duration = max(correct_duration)
    initial_cappings = window + 1
    ending_cappings = 10 - window - 1
    
    longer_than = max(0, min_duration - window)   # duration cannot be negative
    shorter_than = min(10, max_duration + window + 1) ## duration cannot exceed 10
    
    shorter_than_str = duration_dict[shorter_than]
    longer_than_str = duration_dict[longer_than]
    
    base_string = "" + add_being(event_pos) + full_event_text + " "
    
    if min_duration < initial_cappings:
        final_string = base_string + modal(shorter_than) + " shorter than"+  \
                        article(shorter_than) + shorter_than_str + "."
        return [final_string.capitalize()]
    
    elif max_duration > ending_cappings:
        final_string = base_string + modal(longer_than)  + " longer than" +  \
                        article(longer_than) + longer_than_str + "."
        return [final_string.capitalize()]
    else:
        entailed1 = modal(longer_than) + " longer than" + article(longer_than)  + longer_than_str + "."
        entailed2 = modal(shorter_than) + " shorter than"+ article(shorter_than) + shorter_than_str + "."

        return [(base_string + entailed1).capitalize(), 
                (base_string + entailed2).capitalize()]
    
def duration_hypothesis_notentailed(event_id, 
                                 event_text, 
                                 event_pos,
                                 event_lemma,
                                 correct_duration,
                                 ewt,
                                 window=1,
                                 dobj_bool=False):
    '''
    Based on the sliding window (default = 1), generate entailed
    hypothesis
    
    correct_duration = List of duration labels for the given event
                        (list has only 1 item for train, 3 for devtest)
    
    '''
    ## Create full event_text based on dependencies of the predicate
    full_event_text = create_full_event_text(event_id, 
                                             event_text, 
                                             event_pos, 
                                             event_lemma,
                                             ewt,
                                             dobj_bool=dobj_bool)

    ## Duration calculation
    min_duration = min(correct_duration)
    max_duration = max(correct_duration)
    initial_cappings = window + 1
    ending_cappings = 10 - window - 1
    
    longer_than = max(0, min_duration - window)   # duration cannot be negative
    shorter_than = min(10, max_duration + window + 1) ## duration cannot exceed 10
    
    shorter_than_str = duration_dict[shorter_than]
    longer_than_str = duration_dict[longer_than]
    
    base_string = "" + add_being(event_pos) + full_event_text + " "
    
    if min_duration < initial_cappings:
        final_string = base_string + modal(shorter_than) + " longer than"+  \
                        article(shorter_than) + shorter_than_str + "."
        return [final_string.capitalize()]
    
    elif max_duration > ending_cappings:
        final_string = base_string + modal(longer_than)  + " shorter than" + \
                        article(longer_than) + longer_than_str + "."
        return [final_string.capitalize()]
    else:
        notentailed1 = modal(longer_than) + " shorter than" +  article(longer_than)  + longer_than_str + "."
        notentailed2 = modal(shorter_than) + " longer than" +  article(shorter_than)  + shorter_than_str + "."
        return  [(base_string + notentailed1).capitalize(), 
                (base_string + notentailed2).capitalize()]


def create_duration_NLI(event_pair_id, 
                        data,
                        ewt,
                        pairid = 0, 
                        skipcount = 0,
                        event = 1, 
                        sliding_window=1,
                       extra_info = False):
    '''
    Inputs
    1. An event-pair ID from UDS-time 
    
    2. data: UDS-Time dataframe (processed version with normalized)
    
    3. ewt: Corpus class UDS-Time from doc_utils
    
    4. pairid : current pair id (to be incremented in the function)
    
    5. skipcount: count the number of pairs skipped in the process
    
    6. event: 1 for first event, 2 for second event in the pair
    
    7. sliding window: +/- n rank difference 
    
    Outputs:
    A list of dicts:
    extract premise, hypothesis, with labels: entailed and not entailed
    and correct annotations . 
    
    '''
    row = data[data['Event.Pair.ID']==event_pair_id]
    
    recasted_data = []
    recasted_metadata = []
    
    combined_sent_id = getattr(row, 'Combined.Sent.ID').values[0]
    premise = doc_utils.uds_t_sentence_from_id(combined_sent_id, ewt, item="form")
    split = getattr(row, 'Split').values[0]
    
    ## Predicate Information
    pred_text_original = getattr(row, 'Pred' + str(event)+ '.Text.Full').values[0]
    pred_duration = getattr(row, 'Pred' + str(event)+ '.Duration').values
    pred_lemma = getattr(row, 'Pred' + str(event)+ '.Lemma').values[0]
    pred_upos = getattr(row, 'Pred' + str(event)+ '.UPOS').values[0]
    event_id =  getattr(row, 'Event' + str(event)+ '.ID').values[0]
    sentid, tokenid = event_id.split("_")
    #pred_root_word = ewt.decomp_graph[doc_utils.conllu_to_decomp(sentid)].sentence.split()[int(tokenid)]
    
    ## Skip sentence if event_text appears more than once in the sentence to avoid confusion.
    combined_sent_lemmas = doc_utils.uds_t_sentence_from_id(combined_sent_id, 
                                 ewt, tokens=True, 
                                 item="lemma")

    ## If the predicate lemma is present more than once, make the direct object boolean true
    ## dobj_bool is used later in 
    if combined_sent_lemmas.count(pred_lemma) > 1:
        dobj_bool = True
    else:
        dobj_bool = False
    
    ## Skip Eventid if event_upos in 'DET' or 'AUX'
    if pred_upos in ['AUX', 'DET']:
        skipcount+=1
        return 0, 0, pairid, skipcount 
    
    ## Pred NLI:
    entailed_hypothesis = duration_hypothesis_entailed(event_id, 
                                                       pred_text_original, 
                                                       pred_upos,
                                                       pred_lemma,
                                                       pred_duration, 
                                                       ewt,
                                                       window=sliding_window,
                                                       dobj_bool=dobj_bool)
    
    notentailed_hypothesis = duration_hypothesis_notentailed(event_id, 
                                                           pred_text_original, 
                                                           pred_upos,
                                                           pred_lemma,
                                                           pred_duration, 
                                                           ewt,
                                                           window=sliding_window,
                                                           dobj_bool=dobj_bool)
    # Entailed Data
    for curr_hypothesis in entailed_hypothesis:
        pairid+=1
        temp_dict = {}
        temp_dict['context'] = premise
        temp_dict['hypothesis'] = modify_string(curr_hypothesis)
        temp_dict['label'] = 'entailed'
        temp_dict['pair-id'] = pairid
        temp_dict['split'] = split
        temp_dict['type-of-inference'] = 'temporal-duration'
        
        if extra_info:
            temp_dict['raw_predicate'] = pred_text_original
            temp_dict['predicate_lemma'] = pred_lemma
            temp_dict['predicate_upos'] = pred_upos
            temp_dict['corpus-sent-id'] = event_pair_id + "|event_" + str(event)
            
        ##Metadata
        metadata_dict = {}
        metadata_dict['corpus'] = 'uds-time'
        metadata_dict['corpus-license'] = 'todo'
        metadata_dict['corpus-sent-id'] = event_pair_id + "|event_" + str(event)
        metadata_dict['creation-approach'] = 'automatic'
        metadata_dict['pair-id'] = pairid
        
        recasted_data.append(temp_dict)
        recasted_metadata.append(metadata_dict)
    
    # Not entailed
    for curr_hypothesis in notentailed_hypothesis:
        pairid+=1
        temp_dict = {}
        temp_dict['context'] = premise
        temp_dict['hypothesis'] = modify_string(curr_hypothesis)
        temp_dict['label'] = 'not-entailed'
        temp_dict['pair-id'] = pairid
        temp_dict['split'] = split
        temp_dict['type-of-inference'] = 'temporal-duration'
        
        if extra_info:
            temp_dict['raw_predicate'] = pred_text_original
            temp_dict['predicate_lemma'] = pred_lemma
            temp_dict['predicate_upos'] = pred_upos
            temp_dict['corpus-sent-id'] = event_pair_id + "|event_" + str(event)
        
        recasted_data.append(temp_dict)
        
        metadata_dict = {}
        metadata_dict['corpus'] = 'uds-time'
        metadata_dict['corpus-license'] = 'todo'
        metadata_dict['corpus-sent-id'] = event_pair_id + "|event_" + str(event)
        metadata_dict['creation-approach'] = 'automatic'
        metadata_dict['pair-id'] = pairid
        recasted_metadata.append(metadata_dict)
        
    return recasted_data, recasted_metadata, pairid, skipcount


# ## Recast Train Data

def main():
        # Data Locations
    parser = argparse.ArgumentParser(
        description='Recast UDS-Time duration to NLI format.')
    parser.add_argument('--udstime', type=str,
                        default='time_eng_ud_v1.2_2015_10_30.tsv',
                        help='UDS-Time tsv dataset file location.')

    parser.add_argument('--split', type=str,
                        default='',
                        help='If specified (train, dev, test), only that split is recasted')

    parser.add_argument('--out_train', type=str,
                        default='train/',
                        help='recasted train data folder location ')

    parser.add_argument('--out_dev', type=str,
                        default='dev/',
                        help='recasted train data folder location')

    parser.add_argument('--out_test', type=str,
                        default='test/',
                        help='recasted train data folder location ')

    args = parser.parse_args()


    # ### Import UDS Time
    uds_time = pd.read_csv(args.udstime, sep="\t")
    ewt = doc_utils.Corpus(uds_time=uds_time)
    df = ewt.process_data

    #######################################################
    ## Add features to UDS-time dataframe
    #######################################################

    df['Pred1.UPOS'] = df.apply(lambda row: get_predicate_pos(row, ewt, event=1), axis=1)
    df['Pred2.UPOS'] = df.apply(lambda row: get_predicate_pos(row, ewt, event=2), axis=1)

    ## Extract Predicate Full Text
    predicate_dict = {}
    for ud_data_path in ud_data:
        covered_set = set()
        fname = ud_data_path.split("/")[-1]
        data_name = fname.split(".")[0].split("-")[-1]
        
        #print(f"Start processing: {data_name}")
        with open(ud_data_path) as infile:
            data = infile.read()
            parsed = [(PredPatt(ud_parse, opts=options), sent_id) for sent_id, ud_parse in load_conllu(data)]
        
        for pred_object, sentid in parsed:
            sentnum = sentid.split("_")[-1]
            sentenceid = fname + " " + sentnum
            for predicate_object in pred_object.instances:
                #print(f"sentenceid: {sentenceid}, pred: {predicate_object}")
                pred_text, _, pred_root_token,_ = predicate_info(predicate_object)
                predicate_dict[sentenceid + "_" + str(pred_root_token)]= pred_text
                #print(f"error at sentid :{sentenceid}")
                
        print(f"Finished creating predicate dictionary for : {data_name}\n")

    df['Pred1.Text.Full'] = df['Event1.ID'].map(lambda x: predicate_dict[x])
    df['Pred2.Text.Full'] = df['Event2.ID'].map(lambda x: predicate_dict[x])

    #######################################################
    ## Recast Data
    #######################################################
    
    pairid = -1  # count total pair ids
    # Count event-pairs skipped due to ambiguous text for highlighting predicate.
    skipcount = 0

    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'dev', 'test']

    for split in splits:
        data = []
        metadata = []

        curr_df = df[df['Split']==split]
        print(f"Creating NLI instances for Data split: {split}")
        event_pair_ids = list(curr_df.groupby(['Event.Pair.ID']).groups.keys())

        pbar = tqdm(total = len(event_pair_ids))

        for idx, event_pair_id in enumerate(event_pair_ids):
        	## Predicate 1

            recasted_data, recasted_metadata, pairid, skipcount = create_duration_NLI(event_pair_id, df,
                                                                                      ewt, pairid=pairid,
                                                                                      skipcount=skipcount,
                                                                                      event=1,
                                                                                  sliding_window=1)
            if recasted_data:
                data += recasted_data
                metadata += recasted_metadata
            ## Predicate 2
            recasted_data, recasted_metadata, pairid, skipcount = create_duration_NLI(event_pair_id, df, 
                                                                                      ewt, pairid=pairid,
                                                                                      skipcount=skipcount,
                                                                                      event=2,
                                                                                      sliding_window=1)
            if recasted_data:
                data += recasted_data
                metadata += recasted_metadata

            # if pairid%10000==0:
            # 	print(f"Total pair-ids processed so far: {pairid}, skipped so far: {skipcount}")
            pbar.update(1)
            
        out_folder = {'train': args.out_train, 'dev':args.out_dev, 'test':args.out_test}

        print(f"Total pair-ids processed so far: {pairid}, skipped so far: {skipcount}")

        with open(out_folder[split] + "recast_temporal-duration_data.json", 'w') as out_data:
            json.dump(data, out_data, indent=4)

        with open(out_folder[split] + "recast_temporal-duration_metadata.json", 'w') as out_metadata:
            json.dump(metadata, out_metadata, indent=4)


    print(f"Total pair-ids: {pairid}")
    print(f'Total events skipped: {skipcount}')

if __name__== "__main__":
	main()
