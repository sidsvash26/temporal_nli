'''
Author: Siddharth Vashishtha
Affiliation: University of Rochester
Creation Date: 23th May, 2020

Process TimeBank-Dense Data files and convert into a pandas dataframe

Usage: python data_loader_timebank_dense.py
'''

from data_loader_timebank import *

tb_location = "../../data/timebank_data/TBAQ-cleaned/TimeBank/"
aq_location = "../../data/timebank_data/TBAQ-cleaned/AQUAINT/"
tempeval3_test = "../../data/timebank_data/te3-platinum/"
tb_dense_file_loc = "../../data/timebank_data/timebank-dense/TimebankDense.T3.txt"

def timebank_extract_doc_info(folder):
	'''
	Input: Timebank Folder containing xml files

	Output: A dict with keys as document-file-names, and values as a list of multiple info in them

	'''
	doc_info = {}
	for file_path in glob.glob(folder + "*.tml"):    
	    with codecs.open(file_path, 'r') as f:
	        filename = file_path.split("/")[-1]
	        print("File processing: {}".format(filename))
	        xml_str = f.read()
	        xmlSoup = BeautifulSoup(xml_str, 'xml')
	        doc_id = str(xmlSoup.find_all('DOCID')[0].contents[0])
	        print(f"docid: {doc_id}")
	        print("\n")

	        try:
	            extrainfo = str(xmlSoup.find_all('EXTRAINFO')[0].contents[0].strip().split(" =")[0])
	        except:
	            extrainfo = "_notfound_"

	        tlinks = xmlSoup.find_all('TLINK')
	        doc_text = extract_text_from_doc(xmlSoup)

	        stanza_doc = stanza_nlp(doc_text)
	        sentences = extract_stanza_sentences(stanza_doc)
	        
	        ## This is only needed for a hack in the function extract_eid_to_sentenceid_tokenid
	        all_combined_tokens_set = set([token for sent in sentences for token in sent.split()])

	        all_eids_dict = extract_all_eids(xmlSoup)
	        all_eids_tuple = list(all_eids_dict.items())

	        eid_to_sent_token_ids = extract_eid_to_sentenceid_tokenid(all_eids_tuple, sentences,
	                                                                 all_combined_tokens_set,
	                                                                 prev_sent_id=0, prev_token_id=0, 
	                                                                ans_dict={})
	        eiid_to_eid_dict = extract_dict_eiid_to_eid(xmlSoup, param="eventID")
	        eid_to_pos = extract_dict_eid_info(xmlSoup, param="pos")
	        pair_dict = defaultdict(list)
	        
	        ## Save document info
	        doc_info[doc_id] = [all_eids_dict, eid_to_sent_token_ids, eid_to_pos, 
	        					sentences, extrainfo, stanza_doc]
	        
	        eid_to_sent_token_ids = {}
	        
	return doc_info

def create_tb_dense_split(docid):
	'''
	docid in timebank dense

	dev_docs: a list of doc names in dev
	test_docs: a list of doc names in test

	## dev and test docs taken from evaluation code of the original paper:
	## https://github.com/nchambers/caevo/blob/master/src/main/java/caevo/Evaluate.java
	'''

	devDocs = { "APW19980227.0487",
	"CNN19980223.1130.0960", "NYT19980212.0019",
	"PRI19980216.2000.0170", "ed980111.1130.0089" }

	testDocs = { "APW19980227.0489",
	"APW19980227.0494", "APW19980308.0201", "APW19980418.0210",
	"CNN19980126.1600.1104", "CNN19980213.2130.0155",
	"NYT19980402.0453", "PRI19980115.2000.0186",
	"PRI19980306.2000.1675" }

	if docid in devDocs:
	    return "dev"

	elif docid in testDocs:
	    return "test"
	else:
	    return "train"


def main():

	## Import TB-Dense Txt file
	tb_dense = pd.read_csv(tb_dense_file_loc, header=None, sep='\t')
	tb_dense.columns = ['doc_id', 'e1', 'e2', 'td_relation']

	## extract only event-event relations
	tb_dense['event'] = tb_dense.apply(lambda row: row.e1[0]==row.e2[0]=="e", axis=1)
	# print(tb_dense.shape)
	tb_dense.head()
	tb_dense = tb_dense[tb_dense.event==True]
	tb_dense = tb_dense.reset_index(drop=True)
	# print(tb_dense.shape)

	tb_docs_info = timebank_extract_doc_info(tb_location)

	## Add Timebank data info into tb-dense
	tbdense_row_dict = defaultdict(list)
	for idx, row in tb_dense.iterrows():
	    docid = row['doc_id']
	    e1 = row['e1']
	    e2 = row['e2']
	    td_relation = row['td_relation']
	    eids_to_pred_text_dict, eid_to_sent_token_ids, eid_to_pos, sentences, extrainfo, stanza_doc = tb_docs_info[docid]
	    tbdense_row_dict[idx].append(docid)
	    tbdense_row_dict[idx].append(extrainfo)
	    tbdense_row_dict[idx].append(e1)
	    tbdense_row_dict[idx].append(e2)
	    tbdense_row_dict[idx].append(td_relation)
	    ## pred_text
	    tbdense_row_dict[idx].append("".join(eids_to_pred_text_dict[e1]))
	    tbdense_row_dict[idx].append("".join(eids_to_pred_text_dict[e2]))
	    ## pred_pos
	    tbdense_row_dict[idx].append(eid_to_pos[e1])
	    tbdense_row_dict[idx].append(eid_to_pos[e2])
	    ##sentids
	    eid1_sentid, eid1_token_id = eid_to_sent_token_ids[e1]
	    eid2_sentid, eid2_token_id = eid_to_sent_token_ids[e2]
	    tbdense_row_dict[idx].append(eid1_sentid)
	    tbdense_row_dict[idx].append(eid1_token_id)
	    tbdense_row_dict[idx].append(eid2_sentid)
	    tbdense_row_dict[idx].append(eid2_token_id)
	    ## sents
	    tbdense_row_dict[idx].append(sentences[eid1_sentid])
	    tbdense_row_dict[idx].append(sentences[eid2_sentid]) 
	    ## sent_conllus
	    tbdense_row_dict[idx].append(extract_stanza_conllu(stanza_doc,eid1_sentid))
	    tbdense_row_dict[idx].append(extract_stanza_conllu(stanza_doc,eid2_sentid)) 


	columns = ['docid', 'extrainfo', 'eventInstanceID', 'relatedToEventInstance', 
				'td_relation', 'eid1_text', 'eid2_text', 'eid1_POS', 'eid2_POS', 
				'eid1_sent_id','eid1_token_id', 'eid2_sent_id','eid2_token_id',
				'eid1_sentence', 'eid2_sentence', 'eid1_sent_conllu', 'eid2_sent_conllu']

	tb_dense_full_data = pd.DataFrame.from_dict(tbdense_row_dict, orient='index',
	                columns = columns)

	tb_dense_full_data['combined_sent'] = tb_dense_full_data.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="sent"),
	                                                     axis=1)
	tb_dense_full_data['combined_tokenid1'] = tb_dense_full_data.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid1"),
	                                         axis=1)
	tb_dense_full_data['combined_tokenid2'] = tb_dense_full_data.apply(lambda row: extract_combined_sentence_or_tokenids(row, param="tokenid2"),
	                                                 axis=1)

	tb_dense_full_data['corpus'] = "tb-dense"


	tb_dense_full_data['split'] = tb_dense_full_data['docid'].map(lambda x: create_tb_dense_split(x))

	tb_dense_full_data.to_csv("timebank-dense-all.csv", index=False)

	# print(f"\nSaving lemma dict for entire tb-dense corpus\n")

	# lemma_dict = {}
	# for idx, row in tb_dense_full_data.iterrows():
	#     if idx%100==0:
	#         print(f"rows processed: {idx}")
	#     delim = "|"
	#     event_pair_id = row.corpus + delim + row.docid + delim + row.eventInstanceID + delim +  row.relatedToEventInstance
	#     stanza_doc = stanza_nlp(row.combined_sent)
	#     lemma_dict[event_pair_id] = extract_stanza_tokens(stanza_doc, param="lemma")

	# save_obj(lemma_dict, "tb-dense-lemma-dict-all.pkl")

if __name__ == '__main__':
	main()
