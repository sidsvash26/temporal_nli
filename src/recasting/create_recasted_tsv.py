'''
This script converts temporal recasting json data to tsv dataframes
and saves them in the data folder

Usage: python create_recasted_tsv.py 
'''

import pandas as pd
import json

locs = {}
locs['train'] = "../../data/train/"
locs['dev'] = "../../data/dev/"
locs['test'] = "../../data/test/"

## Same filename for train, dev and test
duration_data_file = "recast_temporal-duration_data.json"
temporal_data_file = "recast_temporal-relation_data.json"

## Metadata
duration_metadata_file = "recast_temporal-duration_metadata.json"
temporal_metadata_file = "recast_temporal-relation_metadata.json"

def convert_to_tsv(json_filename, output_filename, save=True,
                  metadata=False):
    '''
    Convert a recasted json file to a tsv dataset
    
    this tsv dataset is used later by transfomers processors
    '''
    with open(json_filename) as f1:
        data = json.load(f1)
    df = pd.DataFrame(data)
    df['index'] = df.index
    
    #reorder columns for data
    if not metadata:
        columns = ['index', 'context', 'hypothesis', 
                      'pair-id', 'type-of-inference','split', 
                      'label']
        df = df[columns]
    
    if save:
        df.to_csv(output_filename, sep='\t', index=False)
        print(f"File saved as: {output_filename}")
    
    return None

def main():
    
    #####################
    ###### Data
    #####################
    for split in ['train', 'dev', 'test']:
        convert_to_tsv(locs[split] + duration_data_file, locs[split] + split + '-temporal-duration-data.tsv')
        convert_to_tsv(locs[split] + temporal_data_file, locs[split] + split +  '-temporal-relation-data.tsv')
        convert_to_tsv(locs[split] + "recast_red_data.json", locs[split] + split +  '-red-data.tsv')
        convert_to_tsv(locs[split] + "recast_tbdense_data.json", locs[split] + split +  '-tbdense-data.tsv')
        convert_to_tsv(locs[split] + "recast_tempeval3_data.json", locs[split] + split +  '-tempeval3-data.tsv')
     
    #####################
    ###### Metadata
    #####################
      
        convert_to_tsv(locs[split] + duration_metadata_file, locs[split] + split + '-temporal-duration-metadata.tsv', metadata=True)
        convert_to_tsv(locs[split] + temporal_metadata_file, locs[split] + split +  '-temporal-relation-metadata.tsv', metadata=True)
        convert_to_tsv(locs[split] + "recast_red_metadata.json", locs[split] + split +  '-red-metadata.tsv', metadata=True)
        convert_to_tsv(locs[split] + "recast_tbdense_metadata.json", locs[split] + split +  '-tbdense-metadata.tsv', metadata=True)
        convert_to_tsv(locs[split] + "recast_tempeval3_metadata.json", locs[split] + split +  '-tempeval3-metadata.tsv', metadata=True)
    
if __name__ == '__main__':
    main()

