'''
Author: Siddharth Vashishtha
Affiliation: University of Rochester
Created: May 27, 2020

Usage: python data_loader_red.py
'''

from red_anafora import *

import pandas as pd
import glob
import stanza
import random
import pickle
from recast_utils import *

SEED=42

stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

def save_obj(obj, name):
    with open(name, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def extract_tlinks(rf):
    '''
    Input: rf: redfile object
    
    Output: a list of tlink tuples of size 2:
            tlink_tuple[0]: an edge dict (containing source and target nodes)
            tlink_tuple[1]: relation between the source and target 
            
    An example tuple {'source': ['229@e@PROXY_AFP_ENG_20020105_0162@gold'],
                       'target': ['135@e@PROXY_AFP_ENG_20020105_0162@gold']},
                      'CONTAINS-SUBEVENT')
    '''
    ans = []
    for relid, redrelation in rf.relation_dictionary.items():
        if redrelation.relation_type == 'TLINK':
            edge_box = redrelation.edge_box
            feature_box = redrelation.feature_box
            tlink_tuple = (edge_box, feature_box['type'])
            ans.append(tlink_tuple)
            
    return ans

def extract_eids_objects(rf):
    '''
    rf: redfile object
    
    output: a dictionary of all mention objects (entities, events, timex etc.)
    '''
    return rf.mention_dictionary

def event_info(eventid_obj, base_info="mention_strings"):
    '''
    Input: Event Class object
    Output: [bas_info, span]
    '''
    return [getattr(eventid_obj, base_info), eventid_obj.spanbox]

def extract_stanza_sentences(stanza_doc, info="text", tokens=True):
    '''
    stanza_doc: a stanza document object

    output: a list of info tokens for each sentence
    '''
    sentences = []
    
    if tokens:
        for sent in stanza_doc.sentences:
            curr_sentence = []
            for token in sent.words:
                curr_sentence.append(getattr(token, info))
            sentences.append(curr_sentence)
    else:
        for sent in stanza_doc.sentences:
            curr_sentence = []
            for token in sent.words:
                curr_sentence.append(getattr(token, info))
            sentences.append(" ".join(curr_sentence))
            
    return sentences


def extract_sentence_id_and_tokenid(span_ids, stanza_doc):
    '''
    input: span_ids -> List [start_span, end_span]
    
    Take the stanza token whose start_span and end_spans match
    with the input start_span
    '''
    gold_start_span, gold_end_span = span_ids
    #print(f"span_ids: {span_ids}")
    sentences = [sentence for sentence in stanza_doc.sentences]
    
    tokenid_start=""
    
    
    for sent_id,sentence in enumerate(sentences):
        for token_id,token in enumerate(sentence.words):
            span_text_start, span_text_end = token.misc.split("|")
            curr_start_span = span_text_start.split("=")[-1]
            curr_end_span = span_text_end.split("=")[-1]
            #print(f"curr_start_span: {curr_start_span}")
            #print(f"curr_end_span: {curr_end_span}\n")
            ## '>' sign is used for cases where stanza token is misaligned
            ## doc example: pilot_0f03cc5a508d630c6c8c8c61396e31a9
            
            if int(curr_start_span)>=int(gold_start_span):
                tokenid_start = token_id
                
            ## doc example: "pilot_90e2a980c22e41e2b25666f676458343"
            if int(curr_end_span)>=int(gold_end_span):
                tokenid_end = token_id
                ## in case the token is split between spans
                ## doc example: pilot_0f03cc5a508d630c6c8c8c61396e31a9
                if tokenid_start=="":
                    tokenid_start=tokenid_end
                return sent_id,list(range(tokenid_start,tokenid_end+1))
            
    return print("sent id and token id not found")


def extract_tlink_dicts(tlinks, eids, stanza_doc):
    '''
    extract a list of dicts with tlinks info    
    
    inputs: 1. tlinks: output of extract_tlinks
            2. eids: output of extract_eids_objects
            3. stanza document object processed on raw text
    '''
    ans_dict = []
    
    ## Doc level info
    sentences_tokens = extract_stanza_sentences(stanza_doc, info="text", tokens=True)
    sentences_pos = extract_stanza_sentences(stanza_doc, info="pos", tokens=True)
    sentences_lemmas = extract_stanza_sentences(stanza_doc, info="lemma", tokens=True)
    sentences_text = extract_stanza_sentences(stanza_doc, info="text", tokens=False)
    idx=0
    for tlink_edge, tlink_relation in tlinks:
        #print(f"idx: {idx}")
        idx+=1
        tlink_dct = {}
        eid1 = tlink_edge['source'][0]
        eid2 = tlink_edge['target'][0]
        
        if eids[eid1].entitytype!="EVENT" or eids[eid2].entitytype!="EVENT":
            continue
            
        ## Pred1
        pred_text1, pred_span1 = event_info(eids[eid1])
        pred_text1 = pred_text1[0]
        pred_span1 = pred_span1[0]
        sentence_id1, token_id1 = extract_sentence_id_and_tokenid(pred_span1, stanza_doc)
        
        ################ Debug info  start ################
        #extracted_token1 = "".join([sentences_tokens[sentence_id1][token] for token in token_id1])
        #print(f"Extracted token: {extracted_token1}")
        #print(f"Gold token: {pred_text1}")
        #assert extracted_token1.strip() == pred_text1.strip()
        ################ Debug info  end ################
        
        pred_lemma1 = "_".join([sentences_lemmas[sentence_id1][token] for token in token_id1])
        pred_pos1 = "_".join([sentences_pos[sentence_id1][token] for token in token_id1])
        
        ## Pred2
        pred_text2, pred_span2 = event_info(eids[eid2])
        pred_text2 = pred_text2[0]
        pred_span2 = pred_span2[0]
        sentence_id2, token_id2 = extract_sentence_id_and_tokenid(pred_span2, stanza_doc)
        
        ################ Debug info  start ################
        #extracted_token2 = "".join([sentences_tokens[sentence_id2][token] for token in token_id2])
        #print(f"Extracted token: {extracted_token2}")
        #print(f"Gold token: {pred_text2}")    
        #assert extracted_token2.strip() == pred_text2.strip()
        ################ Debug info  end ################
        
        pred_lemma2 = "_".join([sentences_lemmas[sentence_id2][token] for token in token_id2])
        pred_pos2 = "_".join([sentences_pos[sentence_id2][token] for token in token_id2])
        
        tlink_dct['eventInstanceID'] = eid1.split("@")[1] + eid1.split("@")[0]
        tlink_dct['relatedToEventInstance'] = eid2.split("@")[1] + eid2.split("@")[0]
        tlink_dct['relation'] = tlink_relation
        
        tlink_dct['eid1_text'] = pred_text1
        tlink_dct['eid2_text'] = pred_text2
        
        tlink_dct['eid1_POS'] = pred_pos1
        tlink_dct['eid2_POS'] = pred_pos2

        tlink_dct['eid1_lemma'] = pred_lemma1
        tlink_dct['eid2_lemma'] = pred_lemma2
        
        tlink_dct['eid1_sent_id'] = sentence_id1
        tlink_dct['eid1_token_id'] = "_".join([str(x) for x in token_id1])
    
        tlink_dct['eid2_sent_id'] = sentence_id2
        tlink_dct['eid2_token_id'] = "_".join([str(x) for x in token_id2])
        
        tlink_dct['eid1_sentence'] = sentences_text[sentence_id1]
        tlink_dct['eid2_sentence'] = sentences_text[sentence_id2]
        
        tlink_dct['eid1_sent_conllu'] = extract_stanza_conllu(stanza_doc,sentence_id2)
        tlink_dct['eid2_sent_conllu'] = extract_stanza_conllu(stanza_doc,sentence_id2)
        
        ans_dict.append(tlink_dct)
        
    return ans_dict

def extract_red_dataframe(text_loc, label_loc):
    '''
    text_loc: folder path "source" data
    label_loc: folder path of the "annotation" data
    '''
    full_dct_lst = []
    
    for file_path in glob.glob(text_loc + "*"):
        #print(file_path)
        text_filename = str(file_path).split("/")[-1]
        label_filename = text_filename + ".RED-Relation.gold.completed.xml"
        
        print(f"Processing file: {text_filename}")
    
        label_file_path = label_loc + label_filename
        raw_text_file_path = text_loc + text_filename
        
        rf = RedFile.fromAnafora(label_file_path, raw_text_file_path)
        
        ## Process through stanza for extracting tokens,pos,lemma 
        stanza_doc = stanza_nlp(rf.raw_text)
        
        tlinks = extract_tlinks(rf)
        eids =  extract_eids_objects(rf)
        
        tlink_dcts = extract_tlink_dicts(tlinks, eids, stanza_doc)
        
        for tlink_dct in tlink_dcts:
            tlink_dct['docid'] = text_filename
            tlink_dct['extrainfo'] = "None"
            full_dct_lst.append(tlink_dct)
            
    return pd.DataFrame(full_dct_lst)


def create_red_split(docid, devDocs, testDocs):
	'''
	docid in red

	dev_docs: a list of doc names in dev
	test_docs: a list of doc names in test

	'''

	if docid in devDocs:
	    return "dev"

	elif docid in testDocs:
	    return "test"
	else:
	    return "train"


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

def main():

	folder_names = ['deft', 'pilot', 'source']

	dfs = []

	for folder_name in folder_names:
		text_loc = "../../data/raw_data/red/data/source/" + folder_name + "/"
		label_loc = "../../data/raw_data/red/data/annotation/" + folder_name + "/"
		df = extract_red_dataframe(text_loc, label_loc)
		dfs.append(df)

	red_all_df = pd.concat(dfs, ignore_index=True)
	
	columns = ['docid', 'extrainfo', 'eventInstanceID', 'relatedToEventInstance', 
				'relation', 'eid1_text', 'eid2_text', 'eid1_POS', 'eid2_POS', 
                'eid1_lemma', 'eid2_lemma','eid1_sent_id','eid1_token_id', 'eid2_sent_id','eid2_token_id',
				'eid1_sentence', 'eid2_sentence', 'eid1_sent_conllu', 'eid2_sent_conllu']

	red_all_df = red_all_df[columns]

	#### Add combined sentences ########
	red_all_df['combined_sent'] = red_all_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="sent"),
	                                                     axis=1)
	red_all_df['combined_tokenid1'] = red_all_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid1"),
	                                         axis=1)
	red_all_df['combined_tokenid2'] = red_all_df.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid2"),
	                                                 axis=1)


	## Create train, dev, test splits:
	total_docs = list(red_all_df.docid.unique())
	dev_docs = random.Random(SEED).sample(total_docs,k=int(len(total_docs)*0.1))
	remaining_docs = set(total_docs) - set(dev_docs)
	test_docs = random.Random(SEED).sample(remaining_docs,k=int(len(total_docs)*0.1))
	
	red_all_df['split'] = red_all_df.docid.map(lambda x:create_red_split(x,dev_docs,test_docs))

	red_all_df['corpus'] = "RED"

	## Save corpus
	red_all_df.to_csv("red-all.csv", index=False)


if __name__ == '__main__':
    main()
