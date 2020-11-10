#!/usr/bin/env python
# coding: utf-8
'''
Usage: python data_loader_timebank.py

'''
# In[1]:
import pickle
import pandas as pd
import numpy as np
import json
import stanza

from bs4 import BeautifulSoup
from collections import defaultdict
import glob
import codecs
import re
from functools import reduce
from recast_utils import *
from tqdm import tqdm

# In[2]:
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
# In[3]:

"""
Created on May 19 11:05:57 2020

@author: sidvash
"""
tb_location = "../../data/raw_data/timebank_data/TBAQ-cleaned/TimeBank/"
aq_location = "../../data/raw_data/timebank_data/TBAQ-cleaned/AQUAINT/"
tempeval3_test = "../../data/raw_data/timebank_data/te3-platinum/"


# ## New Functions

# In[4]:

# For saving objects to pickles


def save_obj(obj, name):
    with open(name, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def extract_text_from_doc(xmlSoup):
    '''
    A list of sentences in xmlSoup

    (all eventids are a single item contained in this list)
    '''
    ans = []

    text_soup = xmlSoup.find_all('TEXT')[0].contents

    for item in text_soup:
        if isinstance(item, str):
            ans.append(item)
        else:  # If event or timex instance
            ans.append(" ".join(item.contents))

    return " ".join(ans)


def extract_all_eids(xmlSoup):
    '''
    xml Soup of a timebank file
    '''
    ans_dict = {}

    text_soup = xmlSoup.findAll('TEXT')[0]
    event_soups = xmlSoup.TEXT.findAll("EVENT")

    for soup in event_soups:
        ans_dict[soup.attrs['eid']] = soup.contents

    return ans_dict


def extract_dict_eid_info(xmlSoup, param=None):
    '''
    xml Soup of a timebank file

    param: ['eiid', 'tense', 'aspect', 'polarity', 'pos']
    '''
    ans_dict = {}
    all_instances = xmlSoup.findAll('MAKEINSTANCE')

    for event_instance in all_instances:
        ans_dict[event_instance.attrs['eventID']] = event_instance.attrs[param]

    return ans_dict


def extract_dict_eiid_to_eid(xmlSoup, param=None):
    '''
    xml Soup of a timebank file

    param: ['eventID', 'tense', 'aspect', 'polarity', 'pos']
    '''
    ans_dict = {}
    all_instances = xmlSoup.findAll('MAKEINSTANCE')

    for event_instance in all_instances:
        ans_dict[event_instance.attrs['eiid']] = event_instance.attrs[param]

    return ans_dict


def extract_stanza_sentences(doc):
    '''
    doc: a stanza document object

    output: a list of sentences (strings) in the given document
    '''
    sentences = []
    for sent in doc.sentences:
        curr_sentence = []
        for token in sent.tokens:
            curr_sentence.append(token.text)
        sentences.append(" ".join(curr_sentence))

    return sentences

def extract_stanza_sentences(doc):
    '''
    doc: a stanza document object

    output: a list of sentences (strings) in the given document
    '''
    sentences = []
    for sent in doc.sentences:
        curr_sentence = []
        for token in sent.tokens:
            curr_sentence.append(token.text)
        sentences.append(" ".join(curr_sentence))

    return sentences


# def extract_stanza_tokens(doc, param="text"):
#     '''
#     doc: a stanza doc object --> assuming this is a single sentence,

#     Output: a list of tokens in the doc
#     '''
#     return [getattr(token, param) for sent in doc.sentences for token in sent.words]


def split_dashes(token_string, delim="-"):
    '''
    Input: earlier-announced-function

    Output: ['earlier', '-', 'announced', '-', 'function']
    '''
    splits = token_string.split(delim)
    ans = []
    for i in range(len(splits)):
        if i != len(splits)-1:
            ans += [splits[i]] + [delim]
        else:
            ans += [splits[i]]
    return ans


def extract_eid_to_sentenceid_tokenid(all_eids_preds, sentences, all_combined_tokens_set,
                                      prev_sent_id=0, prev_token_id=0,
                                      ans_dict={}):
    '''
    all_eids_preds: a dict of all eventids with values as the text
            These are in the order of appearance in the document

    sentences: a list of all sentences (string) in the doc as processed by stanza

    Output:
    dict: key: eid
         value: (sentence_id, token_id)
    '''
    if not all_eids_preds:
        return ans_dict

    eid, event_tokens_orig = all_eids_preds[0]

    ################################
    # This is a cheap HACK!!
    # the following lines take care of multiple span events like "cease-fire"
    # This is tricky coz some cases are split up but some are not like -> "buy-back"

    event_tokens_orig = [x.strip() for x in event_tokens_orig]

    if len(event_tokens_orig) == 1 and ("-" in event_tokens_orig[0]) and (event_tokens_orig[0] not in all_combined_tokens_set):
        event_tokens = split_dashes(event_tokens_orig[0])
    # added exception for one doc in platinum
    elif (event_tokens_orig[0] == "$250"):
        event_tokens = ["$", "250"]
    else:
        event_tokens = event_tokens_orig
    ################################

    len_event_tokens = len(event_tokens)
    curr_sent_id = prev_sent_id
    curr_token_id = prev_token_id

    while curr_sent_id != len(sentences):

        sent_tokens = sentences[curr_sent_id].split()
        len_sent_tokens = len(sent_tokens)
        #print(f"len_sent_tokens: {len_sent_tokens}")

        while curr_token_id != len(sent_tokens):

            # if event id matches
            if event_tokens != sent_tokens[curr_token_id:curr_token_id+len_event_tokens]:
                #print(f"doesn't match: {event_tokens}!= {sent_tokens[curr_token_id:curr_token_id+len_event_tokens]}")
                curr_token_id += 1
                if curr_token_id == len(sent_tokens):
                    curr_sent_id += 1

            else:
                #print(f"match ---> {event_tokens}== {sent_tokens[curr_token_id:curr_token_id+len_event_tokens]}")
                tokenid_span = list(
                    range(curr_token_id, curr_token_id + len_event_tokens))
                #print(f"tokenid_span: {tokenid_span}")
                ans_dict[eid] = (curr_sent_id, "_".join([str(x)
                                                         for x in tokenid_span]))
                # print(ans_dict)

                curr_token_id += len_event_tokens

                return extract_eid_to_sentenceid_tokenid(all_eids_preds[1:], sentences,
                                                         all_combined_tokens_set,
                                                         curr_sent_id, curr_token_id,
                                                         ans_dict)

        curr_token_id = 0

    return extract_eid_to_sentenceid_tokenid(all_eids_preds[1:], sentences,
                                             all_combined_tokens_set,
                                             curr_sent_id, curr_token_id,
                                             ans_dict)


def extract_combined_sentence_or_tokenids(row, param="sent"):
    '''
    param = ['sent', 'tokenid1', 'tokenid2']
    '''

    sentid1 = row["eid1_sent_id"]
    sentid2 = row["eid2_sent_id"]

    tokenid1_org = row["eid1_token_id"]
    tokenid2_org = row["eid2_token_id"]

    if sentid1 == sentid2:
        combined_sentence = row.eid1_sentence

        if param == "sent":
            return combined_sentence
        elif param == "tokenid1":
            return tokenid1_org
        elif param == "tokenid2":
            return tokenid2_org

    elif sentid1 < sentid2:
        combined_sentence = " ".join(
            row.eid1_sentence.split() + row.eid2_sentence.split())
        tokenid1 = tokenid1_org

        sent1_len = len(row.eid1_sentence.split())
        tokenid2 = "_".join(
            list(map(str, [sent1_len + int(x) for x in tokenid2_org.split("_")])))

        if param == "sent":
            return combined_sentence
        elif param == "tokenid1":
            return tokenid1
        elif param == "tokenid2":
            return tokenid2

    else:
        combined_sentence = " ".join(
            row.eid2_sentence.split() + row.eid1_sentence.split())
        tokenid2 = tokenid2_org

        sent2_len = len(row.eid2_sentence.split())
        tokenid1 = "_".join(
            list(map(str, [sent2_len + int(x) for x in tokenid1_org.split("_")])))

        if param == "sent":
            return combined_sentence
        elif param == "tokenid1":
            return tokenid1
        elif param == "tokenid2":
            return tokenid2


# In[5]:
def read_timebank_folder(folder):
    '''
    Extract timebank data as a pandas dataset
    '''
    num_tlinks, num_event_tlinks = 0, 0
    global_dfs = []
    num_docs = 0

    for file_path in glob.glob(folder + "*.tml"):
        with codecs.open(file_path, 'r') as f:
            print("File processing: {}".format(file_path.split("/")[-1]))
            print("\n")
            xml_str = f.read()
            xmlSoup = BeautifulSoup(xml_str, 'xml')
            doc_id = str(xmlSoup.find_all('DOCID')[0].contents[0])

            try:
                extrainfo = str(xmlSoup.find_all('EXTRAINFO')[
                                0].contents[0].strip().split(" =")[0])
            except:
                extrainfo = "_notfound_"

            tlinks = xmlSoup.find_all('TLINK')
            doc_text = extract_text_from_doc(xmlSoup)

            stanza_doc = stanza_nlp(doc_text)
            sentences = extract_stanza_sentences(stanza_doc)

            # This is only needed for a hack in the function extract_eid_to_sentenceid_tokenid
            all_combined_tokens_set = set(
                [token for sent in sentences for token in sent.split()])

            all_eids_dict = extract_all_eids(xmlSoup)
            all_eids_tuple = list(all_eids_dict.items())

            eid_to_sent_token_ids = extract_eid_to_sentenceid_tokenid(all_eids_tuple, sentences,
                                                                      all_combined_tokens_set)

            eiid_to_eid_dict = extract_dict_eiid_to_eid(
                xmlSoup, param="eventID")
            eid_to_pos = extract_dict_eid_info(xmlSoup, param="pos")

            pair_dict = defaultdict(list)

            for soup in tlinks:
                if "eventInstanceID" in soup.attrs and "relatedToEventInstance" in soup.attrs:
                    eid1, eid2 = eiid_to_eid_dict[soup['eventInstanceID']
                                                  ], eiid_to_eid_dict[soup['relatedToEventInstance']]
                    pair_dict['eventInstanceID'].append(eid1)
                    pair_dict['relatedToEventInstance'].append(eid2)
                    pair_dict['relType'].append(soup['relType'])

                    pair_dict['eid1_text'].append("".join(all_eids_dict[eid1]))
                    pair_dict['eid2_text'].append("".join(all_eids_dict[eid2]))

                    pair_dict['eid1_POS'].append(eid_to_pos[eid1])
                    pair_dict['eid2_POS'].append(eid_to_pos[eid2])

                    eid1_sent_id, eid1_token_id = eid_to_sent_token_ids[eid1]
                    eid2_sent_id, eid2_token_id = eid_to_sent_token_ids[eid2]

                    pair_dict['eid1_sent_id'].append(eid1_sent_id)
                    pair_dict['eid1_token_id'].append(eid1_token_id)

                    pair_dict['eid2_sent_id'].append(eid2_sent_id)
                    pair_dict['eid2_token_id'].append(eid2_token_id)

                    pair_dict['eid1_sentence'].append(sentences[eid1_sent_id])
                    pair_dict['eid2_sentence'].append(sentences[eid2_sent_id])

                    pair_dict['eid1_sent_conllu'].append(extract_stanza_conllu(stanza_doc,eid1_sent_id))
                    pair_dict['eid2_sent_conllu'].append(extract_stanza_conllu(stanza_doc,eid2_sent_id))

            print("creating dfs")
            # Create Dataframes
            curr_df = pd.DataFrame(pair_dict)
            curr_df['docid'] = doc_id
            curr_df['extrainfo'] = extrainfo

            cols = ['docid', 'extrainfo'] + list(curr_df.columns[:-2])

            curr_df = curr_df[cols]

            if curr_df.shape[0]:
                curr_df['combined_sent'] = curr_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="sent"),
                                                         axis=1)
                curr_df['combined_tokenid1'] = curr_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid1"),
                                                             axis=1)
                curr_df['combined_tokenid2'] = curr_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid2"),
                                                             axis=1)

                global_dfs.append(curr_df)

            num_docs += 1

            if num_docs % 5 == 0:
                print("Num of docs processed: {}".format(num_docs))
                print("\n")
            num_tlinks += len(tlinks)
            num_event_tlinks += len(pair_dict['relType'])

            print("Total number of tlinks: {}".format(num_tlinks))
            print("Total number of event-event tlinks: {}".format(num_event_tlinks))

    return pd.concat(global_dfs, ignore_index=True)


# ## Save dataframes

# In[6]:
def main():

    timebank_df = read_timebank_folder(tb_location)
    timebank_df['corpus'] = 'te3-timebank'
    timebank_df.to_csv(tb_location + "timebank_dataframe.csv", index=False)

    # In[7]:
    aq_df = read_timebank_folder(aq_location)
    aq_df['corpus'] = 'te3-aquaint'
    aq_df.to_csv(aq_location + "aq_dataframe.csv", index=False)

    # In[8]:
    tempeval3_test_df = read_timebank_folder(tempeval3_test)
    tempeval3_test_df['corpus'] = 'te3-platinum'
    tempeval3_test_df.to_csv(tempeval3_test + "tempeval3_test_dataframe.csv", 
                            index=False)

    # In[9]:
    te3_all_df = pd.concat([timebank_df, aq_df, 
                            tempeval3_test_df], 
                            ignore_index=True)

    te3_all_df.to_csv("tempeval-3-all.csv", index=False)


if __name__ == '__main__':
    main()
