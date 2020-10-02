
'''
This file includes functions that are 
used by recasting scripts across different
datasets
'''

import pandas as pd
import re
from lemminflect import getInflection #lemminflect: (https://lemminflect.readthedocs.io/en/latest/)
import pickle
from stanza.utils.conll import CoNLL
from nltk import DependencyGraph
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
options = PredPattOpts(resolve_relcl=True, 
                        borrow_arg_for_relcl=True, 
                        resolve_conj=False, 
                        cut=True)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def inflection(pred_lemma, pred_pos, pred_word):
    #print(f"lemma: {pred_lemma}, pos: {pred_pos}, word: {pred_word}")
    if pred_pos=="VERB":
        return getInflection(pred_lemma, tag='VBG')[0]
    else:
        return pred_word

def string_expansions(s):
    s = re.sub(" n't ", " not ", str(s))
    s = re.sub(" nt ", " not ", str(s))
    return s

def unimorph_inflect(lemma: str,
                     pos: str,
                     word: str):
    '''
    Predict the inflected form using Unimorph english data
    If not found in unimoph, back-off to lemminflect
    '''
    if pos!='VERB':
        return word
    else:
        if lemma in lemma_to_inflection_dict:
            return lemma_to_inflection_dict[lemma]
        elif lemma in word_to_lemma_dict:
            if word_to_lemma_dict[lemma] in lemma_to_inflection_dict:
                return lemma_to_inflection_dict[word_to_lemma_dict[lemma]]
            else:
                return inflection(lemma, 'VERB', word)
        else:
            return inflection(lemma, 'VERB', word)


def extract_stanza_conllu(doc, sentid):
    '''
    doc: a stanza doc object for the entire document
    sentid: an integer denoting sentence id in document
            (starts with 0)

    Output: conllu format string output of the sentence
    '''
    input_dict = doc.to_dict()
    conll = CoNLL.convert_dict(input_dict)
    string=""
    for item in conll[sentid]:
        string+=("\t".join(item))
        string+="\n"
    return string

def extract_stanza_info(row, eid_num, param="ctag"):
    '''
    Given a row in pd dataframe, and eid number (1 or 2)
    extract the stanza info from conllu parse

    param: 'ctag', 'lemma', 'word'
    '''
    sent_dict = DependencyGraph(row[f'eid{eid_num}_sent_conllu'], 
                                top_relation_label='root').nodes
    tokenid = int(getattr(row, f'eid{eid_num}_token_id'))
    
    return sent_dict[tokenid+1][param]
    

def extract_all_dependencies(tokenid, dependency_dict):
    '''
    Inputs:
    1. tokenid: tokenid in the dependency_dict whose dependencies are to
                be extracted
    2. dependency_dict: A NLTK dependency nodes dictionary
    
    Output:
    tokenids: Sorted List of a all dependencies (tokenids) of the input tokenid
    '''
    ans = [tokenid]
    #print(f"tokenid: {tokenid}")
    curr_deps = dependency_dict[tokenid]['deps']
    ## Base Case
    if len(curr_deps)==0:
        return ans
    ## Recursive Case
    else:
        for modifer, deep_tokenids in curr_deps.items():
            for deep_tokenid in deep_tokenids:
                ans+= extract_all_dependencies(deep_tokenid, dependency_dict)
        return sorted(ans) 

def find_dependency_tokenids(nodes_dict, tokenid, dobj_bool=False):
    '''
    1. nodes dict (nltk depedency strucuture)
    2. tokenid: a token id to be evaluated on the nodes_dict
                tokenid starts at 0, but nodes_dict starts at 1
                so tokenid+1 is used in the function at various places
    3. dobj_bool: A boolean referring to whether to extract direct object arguments or not
    
    Output:
    return tokenids of all dependencies of the input tokenid (sorted)
    
    '''
    adv_tokenids = []
    adj_tokenids = []
    det_tokenids = []
    neg_tokenids = []
    dobj_tokenids = []
    dobj_modifiers_tokenids = []
    
    #if 'advmod' in nodes_dict[tokenid+1]['deps']:
    #    adv_tokenids = nodes_dict[tokenid+1]['deps']['advmod']
        
    ## extract adjective modifiers of tokenid's tag from nodes_dict
    if 'amod' in nodes_dict[tokenid+1]['deps']:
        adj_tokenids = nodes_dict[tokenid+1]['deps']['amod']
        #adj_words = [nodes_dict[iden]['word'] for iden in adj_tokenids]
        #print(f"AMOD of predicate:{nodes_dict[tokenid+1]['word']}, {adj_words}")
        
    ## extract pos of tokenid's tag from nodes_dict
    token_pos = nodes_dict[tokenid+1]['ctag']
    
    ## Extract determiner modifiers 
    # if token_pos!="NOUN":  ## exclude noun predicates
    if 'det' in nodes_dict[tokenid+1]['deps']:
        det_tokenids = nodes_dict[tokenid+1]['deps']['det']
        #det_words = [nodes_dict[iden]['word'] for iden in det_tokenids]
        #print(f"DET of predicate(not NOUN):{nodes_dict[tokenid+1]['word']},  {det_words}")

    ## Extract negation modifiers
    if 'neg' in nodes_dict[tokenid+1]['deps']:
        neg_tokenids = nodes_dict[tokenid+1]['deps']['neg']

    ## If DOBJ Bool is True
    if dobj_bool:
        ## Extract the first DOBJ modifier only in case of a VERB predicate (don't take PRON dobjs)
        if token_pos=="VERB":
            if 'dobj' in nodes_dict[tokenid+1]['deps']:
                ## Only Take the first direct object and make it a list
                dobj_tokenids = [nodes_dict[tokenid+1]['deps']['dobj'][0]]
                
                ## Remove dobj if it is a pronoun
                if nodes_dict[dobj_tokenids[0]]['ctag'] == "PRON":
                    dobj_tokenids = []
                    
                #dobj_words = [nodes_dict[iden]['word'] for iden in dobj_tokenids]
                #print(f"DOBJ of predicate(only VERB): {nodes_dict[tokenid+1]['word']}, {dobj_words}")
              
        ## Extract modifiers of the DOBJ recursively
        if dobj_tokenids:
            ## last Dobject: dobj_tokenids[-1], --> Assuming there is only one dobj TokenID
            ## Extract all dependencies of the direct object
            dobj_modifiers_tokenids = extract_all_dependencies(dobj_tokenids[-1], nodes_dict)
        
            #dobj_modif_words = [nodes_dict[iden]['word'] for iden in dobj_modifiers_tokenids]
            #print(f"DOBJ_MODIFs:  dobj: {dobj_words}, mods: {dobj_modif_words}")
    
    ## add 1 because dependency nodes dict starts tokenid with 1 instead of 0
    ## also include the tokenid of the input
    out_tokenids = [x-1 for x in adv_tokenids + adj_tokenids + det_tokenids + neg_tokenids+ 
                                dobj_tokenids + dobj_modifiers_tokenids] + [tokenid]
    
    ## ensure duplicates are removed 
    
    return sorted(list(set(out_tokenids)))  

def predicate_info(predicate):
    '''
    Input: Predpatt predicate object 
    Output: pred_text, span_tokenids, root_tokenid, predicate_root_POS
    
    Note: If predicate is copular: pred_text is only upto first 5 words
    '''       

    def _copula_predicate_tokenids(span_tokenids, root_tokenid):
        '''    
        Extract tokenids upto the root predicate

        Inputs: span_tokenids: a list of tokenids of a predicate
                root_tokenid: tokenid of root of a predicate ( a positive integer)
        '''
        sorted_span = sorted(span_tokenids)
        if root_tokenid in sorted_span:
            root_index = sorted_span.index(root_tokenid)
            return sorted_span[:root_index+1]
        else:
            return sorted_span


    #Extend predicate to start from the copula
    if predicate.root.tag not in ["VERB", "AUX"]:
        all_pred = predicate.tokens
        gov_rels = [tok.gov_rel for tok in all_pred]
        if 'cop' in gov_rels:
            #print("cop found")
            cop_pos = gov_rels.index('cop')
            pred_token = [x.position for x in all_pred[cop_pos:]]
            def_pred_token = predicate.root.position  #needed for it_happen set
            ## Keep the span until the root tokenid
            pred_token = _copula_predicate_tokenids(pred_token, def_pred_token)

            ## remove the pronoun from the predicate text (for recasting)
            if predicate.root.tag == "PRON":
                if def_pred_token in pred_token:
                    pred_token.remove(def_pred_token)

            pred = [item.text for item in all_pred if item.position in pred_token]
            cop_bool = True  
        elif predicate.root.tag == "ADJ":
            pred_token = [predicate.root.position]
            pred = [predicate.root.text]
            def_pred_token = predicate.root.position
        else:
            #print("Incompatible predicate found")
            pred_token = []
            pred = ["_nan_"]
            def_pred_token = "_nan_"
            
    #Else keep the root        
    else:
        pred_token = [predicate.root.position]
        pred = [predicate.root.text]
        def_pred_token = predicate.root.position 

    #Stringify pred and pred_tokens:
    #pred_token = "_".join(map(str, pred_token))

    ## Add entire text of predicate
    pred = " ".join(pred)
        
    return pred, pred_token, def_pred_token, predicate.root.tag
#####################################
## TB+ related function:
#####################################

def extract_predpatt_text(row, eid_num:int):
    '''
    Given a pandas dataframe of TB data
    and eid_num (1 or 2)
    
    output predpatt predicate text
    (adds copula fillers in text)
    '''
    tokenid = getattr(row, f'eid{eid_num}_token_id')
    conllu_string = getattr(row, f'eid{eid_num}_sent_conllu')
    parsed_tb = [PredPatt(ud_parse, opts=options) for sent_id, ud_parse in load_conllu(conllu_string)]
    pred_objects = parsed_tb[0].instances
    
    curr_text = getattr(row, f'eid{eid_num}_text')
    
    pred_match = False
    #print(f"{(row['docid'], row['eventInstanceID'], row['relatedToEventInstance'])}")
    if pred_objects:
        for pred in pred_objects:
            if int(pred.root.position)==int(tokenid):
                pred_match = True
                pred_object = pred
                break
            else:
                pred_match=False
        
        if pred_match:
            pred_text, _, _, _ = predicate_info(pred_object)
            return pred_text
        else:
            return curr_text

    else:
        return getattr(row, f'eid{eid_num}_text')


def extract_node_info(node_dict, info="lemma"):
    '''
    Input: NLTK Dependency graph node dict
    
    Output: a list of 'info' tokens for entire sentence 
    '''
    len_sent = len(node_dict)-1
    
    return [node_dict[i][info] for i in range(1,len_sent+1)]
        
def combined_sentence_lemma(sent1_conllu,
                           sent2_conllu,
                           sent1_id,
                           sent2_id):
    sent1_dict = DependencyGraph(sent1_conllu, 
                                top_relation_label='root').nodes
    sent2_dict = DependencyGraph(sent2_conllu, 
                                top_relation_label='root').nodes
    
    if sent1_id==sent2_id:
        return extract_node_info(sent1_dict, info="lemma")
    elif sent1_id < sent2_id:
        return extract_node_info(sent1_dict, info="lemma") + \
                extract_node_info(sent2_dict, info="lemma")
    else:
        return extract_node_info(sent2_dict, info="lemma") + \
                extract_node_info(sent1_dict, info="lemma")
    
def fill_missing_POS(row, eid_num):
    '''
    If the pos tag in row is missing,
    return stanza pos tag 
    
    eid_num: 1 or 2
    '''
    pos = getattr(row, f'eid{eid_num}_POS')
    if pos=="OTHER":
        return getattr(row, f'eid{eid_num}_stanza_POS')
    else:
        return pos   

def add_being(pred_pos):
    if pred_pos=="VERB":
        return "the "
    else:
        return "the being "

def relation_hypothesis(full_event_text1,
                        full_event_text2,
                        pred1_pos,
                        pred2_pos
                        ):
    '''
    given a row of te3 dataframe, construct 8 different hypothesis possible
    '''
     ##### Combined hypothesis  ######
    ans = []
    relation_text1 = [" started before ", " ended before "]
    relation_text2 = [' started.', ' ended.']
    
    ## Order: X, Y
    for item1 in relation_text1:
        for item2 in relation_text2:
            ans.append(add_being(pred1_pos) + full_event_text1 + item1 +
                      add_being(pred2_pos) + full_event_text2 + item2)
    ## Order: Y, X
    for item1 in relation_text1:
        for item2 in relation_text2:
            ans.append(add_being(pred2_pos) + full_event_text2 + item1 +
                      add_being(pred1_pos) + full_event_text1 + item2)
    
    return [string.capitalize() for string in ans]

def create_full_event_text_tb(event_tokenid,
                              event_text,
                              event_pos,
                              event_lemma,
                              dependency_nodes,
                              sentence_tokens,
                              dobj_bool=False):
    '''
    Output: create event_full_text based on dependencies, pos, and inflections

    This is only for timebank related datasets (UDST uses a different function)
    '''
    dependency_tokenids = find_dependency_tokenids(dependency_nodes, int(event_tokenid),
                                                    dobj_bool=dobj_bool)     
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
        if dep_tokenid != int(event_tokenid):
            #print(event_id)
            full_event_text.append(sentence_tokens[dep_tokenid])
        else:
            full_event_text.append(event_text_inflected)
            
    ## In case of copula, return the text after copula
    if len(event_text.split()) > 1:
        return " ".join(event_text.split()[1:])
    
    else:
        return " ".join(full_event_text)

word_to_lemma_dict = load_obj("word_to_lemma_dict.pkl")
lemma_to_inflection_dict= load_obj("lemma_to_inflection_dict.pkl")