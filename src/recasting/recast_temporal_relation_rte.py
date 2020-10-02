

'''
Dependencies:
1. PredPatt: https://github.com/hltcoe/PredPatt
2. lemminflect: https://pypi.org/project/lemminflect/
3. doc_utils: python file inlcuded in this folder
4. pandas
5. numpy
6. UD_english conllu files that are compatible with PredPatt
7. nltk
8. recast_temporal_duration_rte

Usage: python recast_temporal_relation_rte.py \
            --udstime "../../data/UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv" \
            --out_train "../../data/train/" \
            --out_dev "../../data/dev/" \
            --out_test "../../data/test/" 
'''
from recast_temporal_duration_rte import *
from recast_utils import *

def relation_vector(row):
    '''
    Create a Relation vector from UDS-Time sliders
    
    vector: 8-dimensional vector (8 dimensions are commented below)
    '''
    b1  = getattr(row, 'Norm.b1')
    e1 = getattr(row, 'Norm.e1')
    b2  = getattr(row, 'Norm.b2')
    e2 = getattr(row, 'Norm.e2')
    
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

def relation_hypothesis(event_pair_id,
                        combined_sent_id,
                        pred1_text_raw,
                        pred1_upos,
                        pred1_lemma,
                        pred2_text_raw,
                        pred2_upos,
                        pred2_lemma,
                        ewt):
    '''
    Given items from a row of UDS-Time, 
    generate a list of 8 possible relation hypotheses
    '''
    pred1_id, pred2_id = event_pair_id.split("|")
    combined_sent_lemmas = doc_utils.uds_t_sentence_from_id(combined_sent_id, 
                                 ewt, tokens=True, 
                                 item="lemma")

    ##### Predicate 1 ######
    # Check if predicate ambiguity exists, add DOBJ boolean 
    if combined_sent_lemmas.count(pred1_lemma) > 1:
        dobj_bool_pred1 = True
    else:
        dobj_bool_pred1 = False
    full_event_text1 = create_full_event_text(pred1_id, 
                                             pred1_text_raw, 
                                             pred1_upos, 
                                             pred1_lemma,
                                             ewt,
                                             dobj_bool=dobj_bool_pred1)
    ##### Predicate 2 ######
    # Check if predicate ambiguity exists, add DOBJ boolean 
    if combined_sent_lemmas.count(pred2_lemma) > 1:
        dobj_bool_pred2 = True
    else:
        dobj_bool_pred2 = False
    full_event_text2 = create_full_event_text(pred2_id, 
                                             pred2_text_raw, 
                                             pred2_upos, 
                                             pred2_lemma,
                                             ewt,
                                             dobj_bool=dobj_bool_pred2)
    
    ##### Combined hypothesis  ######
    ans = []
    relation_text1 = [" started before ", " ended before "]
    relation_text2 = [' started.', ' ended.']
    
    ## Order: X, Y
    for item1 in relation_text1:
        for item2 in relation_text2:
            ans.append(add_being(pred1_upos) + full_event_text1 + item1 +
                      add_being(pred2_upos) + full_event_text2 + item2)
    ## Order: Y, X
    for item1 in relation_text1:
        for item2 in relation_text2:
            ans.append(add_being(pred2_upos) + full_event_text2 + item1 +
                      add_being(pred1_upos) + full_event_text1 + item2)
    
    return [string.capitalize() for string in ans]
    
def create_relation_NLI(event_pair_id,
                       data,
                       ewt,
                       relation_label_columns = None,
                        pairid = 0,
                       extra_info=False):
    '''
    Given an event-pair id create NLI pairs
    data: UDS Time dataframe with some extra features
    '''
    
    row = data[data['Event.Pair.ID']==event_pair_id]
    
    recasted_data = []
    recasted_metadata = []
    
    combined_sent_id = getattr(row, 'Combined.Sent.ID').values[0]
    premise = doc_utils.uds_t_sentence_from_id(combined_sent_id, ewt, item="form")
    split = getattr(row, 'Split').values[0]
    
    ## Pred1
    pred1_text_raw = getattr(row, 'Pred1.Text.Full').values[0]
    pred1_lemma = getattr(row, 'Pred1.Lemma').values[0]
    pred1_upos = getattr(row, 'Pred1.UPOS').values[0]

    ## Pred2 
    pred2_text_raw = getattr(row, 'Pred2.Text.Full').values[0]
    pred2_lemma = getattr(row, 'Pred2.Lemma').values[0]
    pred2_upos = getattr(row, 'Pred2.UPOS').values[0]

    ## sum(axis=0) is needed for dev-test, for train, values remain the same
    relation_votes = row[relation_label_columns].values.sum(axis=0)
    
    ## Threshold for how many votes are there for one of the 8 relations
    ## The range of votes is [0,3]
    if split=="train":
        entailment_threshold = 1
    elif split=="dev" or split=="test":
        entailment_threshold = 2
        
    
    hypothesis_list = relation_hypothesis(event_pair_id,
                                            combined_sent_id,
                                            pred1_text_raw,
                                            pred1_upos,
                                            pred1_lemma,
                                            pred2_text_raw,
                                            pred2_upos,
                                            pred2_lemma,
                                            ewt)
                                    
    for hypothesis, relation_vote in zip(hypothesis_list, relation_votes):
        pairid +=1
        temp_dict = {}
        temp_dict['context'] = premise        
        temp_dict['hypothesis'] = modify_string(hypothesis)
        
        if relation_vote >=entailment_threshold:
            temp_dict['label'] = 'entailed'
        else:
            temp_dict['label'] = 'not-entailed'
            
        temp_dict['pair-id'] = pairid
        temp_dict['split'] = split
        temp_dict['type-of-inference'] = 'temporal-relation'

        if extra_info:
            temp_dict['pred1_text_raw'] = pred1_text_raw
            temp_dict['pred1_lemma'] = pred1_lemma
            temp_dict['pred1_upos'] = pred1_upos
            temp_dict['pred2_text_raw'] = pred2_text_raw
            temp_dict['pred2_lemma'] = pred2_lemma
            temp_dict['pred2_upos'] = pred2_upos
            temp_dict['corpus-sent-id'] = event_pair_id

        ##Metadata
        metadata_dict = {}
        metadata_dict['corpus'] = 'uds-time'
        metadata_dict['corpus-license'] = 'todo'
        metadata_dict['corpus-sent-id'] = event_pair_id
        metadata_dict['creation-approach'] = 'automatic'
        metadata_dict['pair-id'] = pairid

        recasted_data.append(temp_dict)
        recasted_metadata.append(metadata_dict)
        
    return recasted_data, recasted_metadata, pairid


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
    print(f"Adding features to UDS-Time for hypothesis generation")
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
                pred_text, _, pred_root_token, _ = predicate_info(predicate_object)
                predicate_dict[sentenceid + "_" + str(pred_root_token)]= pred_text
                #print(f"error at sentid :{sentenceid}")
                
        print(f"Finished creating predicate dictionary for : {data_name}")

    df['Pred1.Text.Full'] = df['Event1.ID'].map(lambda x: predicate_dict[x])
    df['Pred2.Text.Full'] = df['Event2.ID'].map(lambda x: predicate_dict[x])


    df['relation_label'] = df.apply(lambda row: relation_vector(row), axis=1)
    df_temp = df['relation_label'].apply(pd.Series)
    df_temp.columns = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7']
    labels = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7']
    df = pd.concat([df, df_temp], axis=1)

    #######################################################
    ## Recast Data
    #######################################################
    print(f"Relation recasting started.....")
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
            
            recasted_data, recasted_metadata, pairid = create_relation_NLI(event_pair_id, df,
                                                                                      ewt, pairid=pairid,
                                                                                      relation_label_columns =labels,
                                                                                extra_info=False)
            if recasted_data:
                data += recasted_data
                metadata += recasted_metadata
            

            # if pairid%(2**15)==0:
            #     print(f"Total pair-ids processed so far: {pairid}")

            pbar.update(1)
            
        out_folder = {'train': args.out_train, 'dev':args.out_dev, 'test':args.out_test}

        with open(out_folder[split] + "recast_temporal-relation_data.json", 'w') as out_data:
            json.dump(data, out_data, indent=4)

        with open(out_folder[split] + "recast_temporal-relation_metadata.json", 'w') as out_metadata:
            json.dump(metadata, out_metadata, indent=4)

    print(f"Total pair-ids: {pairid}")
    
if __name__== "__main__":
    main()
