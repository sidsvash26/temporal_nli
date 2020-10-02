import numpy as np
from typing import Tuple, List
import pickle 
import decomp
import math
import pandas as pd

decomp_graph = decomp.UDSCorpus()

## For saving objects to pickles
def save_obj(obj, name ):
    with open(name , 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
    
def sort_eventids(lst):
    '''
    Sort based on Event_ids (Sentenced-id _ token-id)
    Example input = ['en-ud-train.conllu 9826_3', 'en-ud-train.conllu 9825_2']
    Example Output: ['en-ud-train.conllu 9825_2', 'en-ud-train.conllu 9826_3']
    '''
    return sorted(lst, key=lambda x: int(x.split()[-1].split("_")[0]) + \
                      0.001*(int(x.split()[-1].split("_")[1])))

def load_ud_english(fpath):
    """Load a file from the UD English corpus

    Parameters
    ----------
    fpath : str
        Path to UD corpus file ending in .conllu
        
    Output: Returns a list equal to length of total num of docs
            with each item contaiing the first index of sentence
            in that doc
    """
    import os
    import re
    from collections import defaultdict
    n = 1

    fname = os.path.split(fpath)[1]

    parses = defaultdict(list)
    sent_ids = []
    newdoc_ids = []
    
    for l in open(fpath):
        ident = fname+' '+str(n)
        
        if re.match(r'\# newdoc id', l):
            newdoc_ids.append(n)
            #newdoc_ids.append(l.split("=")[-1].strip())
            
        if re.match(r'^\d', l):
            l_split = l.strip().split()
            parses[ident].append(l_split)
        
        elif parses[ident]:
            sent_ids.append(ident)
            n += 1

    return newdoc_ids, len(sent_ids)

def extract_sent_to_docid_dict(path_list):
    '''
    Input: ewt_ud .conllu file path
    
    Output: A dict with keys: sent_id, value: doc_id
    '''
    sent_to_doc_dict = {}
    
    for ud_ewt_path in path_list:
        doc_lst, total_sents = load_ud_english(ud_ewt_path)
        data_name = ud_ewt_path.split("ud-")[-1].split(".")[0]
        string = "en-ud-" + data_name + ".conllu" + " "

        len_doc_lst = len(doc_lst)
        
        for i in range(len_doc_lst-1):
            for sent_idx in range(doc_lst[i], doc_lst[i+1]):
                sent_to_doc_dict[string + str(sent_idx)] = str(i+1)

        #Last document:
        for sent_idx in range(doc_lst[-1], total_sents+1):
            sent_to_doc_dict[string + str(sent_idx)] = str(len_doc_lst)
    
    #convert docs to integers
    return {key: int(value) for key,value in sent_to_doc_dict.items()}  

def get_nearby_sentids(sentence_id, sent_to_docid_dict):
    '''
    Get the previous and next sentence ids of the input id in its document
    
    sentence_id format: 'en-ud-train.conllu 9825'
    '''
    ans = []
    
    curr_doc = sent_to_docid_dict[sentence_id]
    data_prefix, curr_num = sentence_id.split()    
    
    sent_num = int(curr_num)
    prev_num = sent_num-1
    next_num = sent_num+1
    
    try:
        if sent_to_docid_dict[data_prefix+" " +str(prev_num)]==curr_doc:
            ans.append(prev_num)
    except Exception:
        pass
    
    ans.append(sent_num)
    
    try: 
        if sent_to_docid_dict[data_prefix+" " +str(next_num)]==curr_doc:
            ans.append(next_num)
    except Exception:
        pass
    
    return ans

def combine_sentence(tokens):
    '''
    Join tokens with spaces to form a string, 
    but if the last token is a period, then concatenate it 
    with the second last token
    '''
    sent = ''
    if tokens[-1] in {'.', '!', '?', '!.', '!!', '!!!'} :
        sent = " ".join(tokens[:-1]) + tokens[-1]
    else:
        sent = " ".join(tokens)
        
    return sent

def conllu_to_decomp(sent_id, reverse=False):
    '''
    Convert sentence id to different formats:
    Example:
    Input: 'en-ud-train.conllu 418'
    Output: 'ewt-train-418'
    '''
    if not reverse:
        sent_str, num = sent_id.split(" ")
        split = sent_str.split(".")[0].split("-")[-1]
        sent_name = "ewt-" + split + "-" + num
        return sent_name

    else:
        _, split, num = sent_id.split("-")
        sent_name = "en-ud-" + split + ".conllu " + num
        return sent_name
    
def uds_t_sentence_from_id(uds_t_sent_id, ewt, tokens=False, item="form"):
    '''
    Given the uds_time combined sentence id, extract the text sentence

    item can be "form", "lemma" etc. 
    '''
    sents = uds_t_sent_id.split("|")
    if sents[0]==sents[1]:
        if tokens:
            return ewt.uds_syntax(conllu_to_decomp(sents[0]), item=item)
        else:
            return combine_sentence(ewt.uds_syntax(conllu_to_decomp(sents[0]), item=item))
    else:
        if tokens:
            return ewt.uds_syntax(conllu_to_decomp(sents[0]), item=item) + ewt.uds_syntax(conllu_to_decomp(sents[1]), item=item)
            
        else:
            final_sent= combine_sentence(ewt.uds_syntax(conllu_to_decomp(sents[0]),item=item)) + \
                        " " + combine_sentence(ewt.uds_syntax(conllu_to_decomp(sents[1]), item=item))
            return final_sent
        

class Corpus(object):
    ''' 
    A class to extract features/information from UDS Corpus and UDS-Time dataset
    '''
    def __init__(self, 
                 uds_time=None,
                decomp_graph = decomp_graph):
        self.data = uds_time
        self.decomp_graph = decomp_graph

        ##Store pre-processed data:
        self.process_data = self._preprocess_UDS()
        
    def doc_sids(self, docid, split="train"):
        '''
        data: UDS-Time
        docid: docid number 

        Extracts a list of sentence ids from the data for the input document

        '''
        df = self.data[self.data.Split==split]

        sent_ids = np.unique(df[df['Document.ID']==docid][['Sentence1.ID', 
                                                           'Sentence2.ID']].values.flatten())

        #Sort based on sentence ids
        return sorted(sent_ids, key=lambda x: int(x.split()[-1]))
    
    def doc_events(self, docid, split="train"):
        '''
        data: UDS-Time
        docid: docid number 

        Extracts a list of sorted sentence ids from the data for the input document

        '''
        df = self.data[self.data.Split==split]

        sent_ids = np.unique(df[df['Document.ID']==docid][['Event1.ID', 
                                                           'Event2.ID']].values.flatten())
        
        #Sort based on Event_ids (Sentenced-id _ token-id)
        return sorted(sent_ids, key=lambda x: int(x.split()[-1].split("_")[0]) + \
                      0.001*(int(x.split()[-1].split("_")[1])))

    def doc_event_pairs(self, docid, split='train'):
        '''
        data: UDS-Time
        docid: docid number 
            
        Extracts a set of event pairs ids from the data for the input document

        '''
        df = self.data[self.data.Split==split]

        pairs = df[df['Document.ID']==docid][['Event1.ID', 'Event2.ID']].values
        
        #Sort based on Event_ids (Sentenced-id _ token-id)
        return set([event1 + "|" + event2 for event1,event2 in pairs])

    def uds_syntax(self, sent_id, item="form"):
        '''Extract a list of 'item' in the linear order of the sentence
            from the input uds syntax graph
        '''
        return [dct[item] for idx, dct in self.decomp_graph[sent_id].syntax_nodes.items()]
    
    def doc_text(self, docid, split='train'):
        '''Return the entire document text from the UDS EWT data for the given docid.
        '''
        text = ""
        for sent in self.doc_sids(docid, split=split):
            sentid = sent.split()[-1]
            uds_id = "ewt-" +  split + "-" + sentid
            text += " ".join(self.uds_syntax(uds_id, item='form'))
            text+= " "
        return text

    def median_durations(self, doc_id, split="train"):
        '''
        Extracts a dict of event ids with values as floored median duration and median confidence
        
        '''
        duration_dct = {}
        
        for event_id in self.doc_events(doc_id, split=split):
        
            pred1 = self.process_data[self.process_data['Event1.ID']==event_id]['Pred1.Duration'].values
            pred2 = self.process_data[self.process_data['Event2.ID']==event_id]['Pred2.Duration'].values

            pred1_conf = self.process_data[self.process_data['Event1.ID']==event_id]['Pred1.dur.conf.ridit'].values
            pred2_conf = self.process_data[self.process_data['Event2.ID']==event_id]['Pred2.dur.conf.ridit'].values
            
            duration_dct[event_id] = [math.floor(np.median(np.concatenate([pred1, pred2]))), 
                                        np.median(np.concatenate([pred1_conf, pred2_conf]))]
        return duration_dct
    
    def _preprocess_UDS(self):
        '''
        pre-process UDS data:
        1. normalize slider values
        2. ridit confidence scores
        3. extract predicate 2's tokens in the combined sentence
        
        Inputs:
        1. uds_data: Dataframe of UDS-Time
        2. ewt: An instance of the Corpus class defined in this file
        '''
        uds_copy = self.data.copy()  #deep copy
        
        def _normalize_slider(row, param=None):
            b1 = getattr(row, 'Pred1.Beg')
            e1 = getattr(row, 'Pred1.End')
            b2 = getattr(row, 'Pred2.Beg')
            e2 = getattr(row, 'Pred2.End')
            
            min_val = min(b1,e1,b2,e2)
            b1_adj, e1_adj, b2_adj, e2_adj = [b1-min_val, e1-min_val, b2-min_val, e2-min_val]
            max_val = max([b1_adj, e1_adj, b2_adj, e2_adj])
            
            if param=='b1':
                try:
                    return round(b1_adj/max_val,4)
                except:
                    return 0.0
            elif param=='e1':
                try:
                    return round(e1_adj/max_val, 4)
                except:
                    return 0.0
            elif param=='b2':
                try:
                    return round(b2_adj/max_val,4)
                except:
                    return 0.0
            elif param=='e2':
                try:
                    return round(e2_adj/max_val,4)
                except:
                    return 0.0
            else:
                raise Exception('Slider position not specified')
                
        def ridit_score(x):
            x = x.astype(int)
            x_max, x_min = x.max(), x.min()
            counts = x.value_counts()[list(range(x_max+1))].fillna(0)
            dist = counts/counts.sum()
            cumdist = dist.cumsum()

            def _ridit_score(i):
                if i != x_min: 
                    return cumdist[i-1] + dist[i]/2
                else:
                    return dist[i]/2

            return x.apply(_ridit_score)
        
        def _combined_pred2_root(row):
            '''
            Extracting predicate 2's root posisition in the combined sentence
            '''
            sentence_id_1 = getattr(row, 'Sentence1.ID')
            sentence_id_2 = getattr(row, 'Sentence2.ID')
            if sentence_id_1 == sentence_id_2:
                return getattr(row, 'Pred2.Token')
            else:
                sent_str, num = sentence_id_1.split(" ")
                #Example: sent_str = "en-ud-train.conllu 418"
                split = sent_str.split(".")[0].split("-")[-1]
                sent_name = "ewt-" + split + "-" + num
                return len(self.uds_syntax(sent_name)) + getattr(row, 'Pred2.Token')
            
        def _combined_pred2_spans(row):
            '''
            Extracting predicate 2's span positions in the combined sentence
            '''
            sentence_id_1 = getattr(row, 'Sentence1.ID')
            sentence_id_2 = getattr(row, 'Sentence2.ID')
            if sentence_id_1 == sentence_id_2:
                return getattr(row, 'Pred2.Span')
            else:
                sent_str, num = sentence_id_1.split(" ")
                split = sent_str.split(".")[0].split("-")[-1]
                sent_name = "ewt-" + split + "-" + num
                curr_posns = [int(x) for x in getattr(row, 'Pred2.Span').split("_")]
                new_posns = [len(self.uds_syntax(sent_name)) + x for x in curr_posns]
                return "_".join([str(x) for x in new_posns])
            
        def _combined_sentence_id(row):
            sentence_id_1 = getattr(row, 'Sentence1.ID')
            sentence_id_2 = getattr(row, 'Sentence2.ID')
            
            return sentence_id_1 + "|" + sentence_id_2

        def _event_pair_id(row):
            event1 = getattr(row, 'Event1.ID')
            event2 = getattr(row, 'Event2.ID')

            return event1 + "|" + event2
                
        ## Normalize slider values
        uds_copy.loc[:,'Norm.b1'] = uds_copy.apply(lambda row: _normalize_slider(row, param='b1'), axis=1).values
        uds_copy.loc[:,'Norm.e1'] = uds_copy.apply(lambda row: _normalize_slider(row, param='e1'), axis=1).values
        uds_copy.loc[:,'Norm.b2'] = uds_copy.apply(lambda row: _normalize_slider(row, param='b2'), axis=1).values
        uds_copy.loc[:,'Norm.e2'] = uds_copy.apply(lambda row: _normalize_slider(row, param='e2'), axis=1).values
        
        ## Ridit score confidence values
        uds_copy.loc[:,'Relation.conf.ridit'] = uds_copy.groupby('Annotator.ID')['Relation.Confidence'].transform(ridit_score).values
        uds_copy.loc[:,'Pred1.dur.conf.ridit'] = uds_copy.groupby('Annotator.ID')['Pred1.Duration.Confidence'].transform(ridit_score).values
        uds_copy.loc[:,'Pred2.dur.conf.ridit'] = uds_copy.groupby('Annotator.ID')['Pred2.Duration.Confidence'].transform(ridit_score).values
        
        ## Get predicate token positions in the combined sentence 
        ## Note: For predicate 1 the tokens remain the same in the combined sentence
        uds_copy.loc[:,"Pred2.Mod.Token"] = uds_copy.apply(lambda row: _combined_pred2_root(row), axis=1).values
        uds_copy.loc[:,"Pred2.Mod.Span"] = uds_copy.apply(lambda row: _combined_pred2_spans(row), axis=1).values
            
        uds_copy.loc[:,'Combined.Sent.ID'] = uds_copy.apply(lambda row: _combined_sentence_id(row),
                                                           axis=1).values
        uds_copy.loc[:,'Event.Pair.ID'] = uds_copy.apply(lambda row: _event_pair_id(row),
                                                           axis=1).values
        
        
        ## keep token naming convention consistent
        uds_copy.rename(columns={"Pred1.Token": "Pred1.Mod.Token", 
                                 "Pred1.Span": "Pred1.Mod.Span"}, inplace=True)
        
        
        final_columns = ['Split', 'Document.ID', 'Event1.ID', 'Event2.ID', 'Combined.Sent.ID',
                        'Event.Pair.ID','Pred1.Lemma', 'Pred2.Lemma', 
                         'Pred1.Mod.Span', 'Pred2.Mod.Span',
                         'Pred1.Mod.Token', 'Pred2.Mod.Token',
                         'Pred1.Text', 'Pred2.Text',  
                         'Norm.b1', 'Norm.e1', 'Norm.b2', 'Norm.e2',
                         'Pred1.Duration', 'Pred2.Duration',
                         'Relation.conf.ridit','Pred1.dur.conf.ridit','Pred2.dur.conf.ridit'  
                        ]
        
        return uds_copy[final_columns]

