'''
This script extracts two dictionaries from 
the Unimorph Eng data: https://github.com/unimorph/eng 

Usage: python unimorph_dicts.py
'''

import pandas as pd
from tqdm import tqdm 
import pickle

## For saving objects to pickles
def save_obj(obj, name ):
    with open(name , 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def get_unimorph_dicts(unimorph):
    '''
    given unimorph filename
    
    Output: lemma_to_inflection_dict
            word_to_lemma_dict
    '''
    df = pd.read_csv(unimorph, sep='\t', 
                      header=None,
                      names= ['root', 
                                'inflection', 
                                'feature'])
    
    lemmas = df['root'].unique()
    feature = 'V;V.PTCP;PRS'
    
    lemma_to_inflection_dict = {}
    word_to_lemma_dict = {}
    
    for lemma in tqdm(lemmas):
        inflection = df[(df['root']==lemma) &
                        (df['feature']==feature)]['inflection'].any()
        if inflection:
            lemma_to_inflection_dict[lemma] = inflection 
            
        words = df[df['root']==lemma]['inflection'].values
        
        for word in words:
            word_to_lemma_dict[word]=lemma
            
    return lemma_to_inflection_dict, word_to_lemma_dict

def main():
    print(f"Saving english unimorph dicts")

    lemma_to_inflection_dict, word_to_lemma_dict = get_unimorph_dicts('eng')

    save_obj(lemma_to_inflection_dict, "lemma_to_inflection_dict.pkl")
    save_obj(word_to_lemma_dict, "word_to_lemma_dict.pkl")


if __name__ == "__main__":
    main()
