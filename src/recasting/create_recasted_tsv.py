'''
This script converts temporal recasting json data to tsv dataframes
and saves them in the data folder

Usage: python create_recasted_tsv.py 
'''

import pandas as pd
import json

train_location = "../../data/train/"
dev_location = "../../data/dev/"
test_location = "../../data/test/"

## Same filename for dev and test
duration_data_file = "recast_temporal-duration_data.json"
temporal_data_file = "recast_temporal-relation_data.json"

## Metadata
duration_metadata_file = "recast_temporal-duration_metadata.json"
temporal_metadata_file = "recast_temporal-relation_metadata.json"

## File name for train
dur_data_file_split1 = "recast_temporal-duration_data_split1.json"
dur_data_file_split2 = "recast_temporal-duration_data_split2.json"
dur_data_file_split3 = "recast_temporal-duration_data_split3.json"

rel_data_file_split1 = "recast_temporal-relation_data_split1.json"
rel_data_file_split2 = "recast_temporal-relation_data_split2.json"
rel_data_file_split3 = "recast_temporal-relation_data_split3.json"

## Metadata
dur_metadata_file_split1 = "recast_temporal-duration_metadata_split1.json"
dur_metadata_file_split2 = "recast_temporal-duration_metadata_split2.json"
dur_metadata_file_split3 = "recast_temporal-duration_metadata_split3.json"

rel_metadata_file_split1 = "recast_temporal-relation_metadata_split1.json"
rel_metadata_file_split2 = "recast_temporal-relation_metadata_split2.json"
rel_metadata_file_split3 = "recast_temporal-relation_metadata_split3.json"


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
    
    return df

def main():
    
    #####################
    ###### Data
    #####################
   
    ## Dev
    df = convert_to_tsv(dev_location+duration_data_file, dev_location + 'dev-temporal-duration-data.tsv')
    df = convert_to_tsv(dev_location+temporal_data_file, dev_location + 'dev-temporal-relation-data.tsv')
    
    df = convert_to_tsv(dev_location+"recast_red_data.json", dev_location + 'dev-red-data.tsv')
    df = convert_to_tsv(dev_location+"recast_tbdense_data.json", dev_location + 'dev-tbdense-data.tsv')
    df = convert_to_tsv(dev_location+"recast_tempeval3_data.json", dev_location + 'dev-tempeval3-data.tsv')    
    ## Test
    df = convert_to_tsv(test_location+duration_data_file, test_location + 'test-temporal-duration-data.tsv')
    df = convert_to_tsv(test_location+temporal_data_file, test_location + 'test-temporal-relation-data.tsv')
    
    df = convert_to_tsv(test_location+"recast_red_data.json", test_location + 'test-red-data.tsv')
    df = convert_to_tsv(test_location+"recast_tbdense_data.json", test_location + 'test-tbdense-data.tsv')
    df = convert_to_tsv(test_location+"recast_tempeval3_data.json", test_location + 'test-tempeval3-data.tsv')   

 
    ## Train
    df1 = convert_to_tsv(train_location+dur_data_file_split1, None, save=False)
    df2 = convert_to_tsv(train_location+dur_data_file_split2, None, save=False)
    df3 = convert_to_tsv(train_location+dur_data_file_split3, None, save=False)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['index'] = df.index
    df.to_csv(train_location + 'train-temporal-duration-data.tsv', sep='\t', index=False)
    print(f"File saved as: {train_location + 'train-temporal-duration-data.tsv'}")
    
    df1 = convert_to_tsv(train_location+rel_data_file_split1, None, save=False)
    df2 = convert_to_tsv(train_location+rel_data_file_split2, None, save=False)
    df3 = convert_to_tsv(train_location+rel_data_file_split3, None, save=False)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['index'] = df.index
    df.to_csv(train_location + 'train-temporal-relation-data.tsv', sep='\t', index=False)
    print(f"File saved as: {train_location + 'train-temporal-relation-data.tsv'}")
    
    df = convert_to_tsv(train_location+"recast_red_data.json", train_location + 'train-red-data.tsv')
    df = convert_to_tsv(train_location+"recast_tbdense_data.json", train_location + 'train-tbdense-data.tsv')
    df = convert_to_tsv(train_location+"recast_tempeval3_data.json", train_location + 'train-tempeval3-data.tsv')

    #####################
    ###### Metadata
    #####################
    ## Dev
    df = convert_to_tsv(dev_location+duration_metadata_file, dev_location + 'dev-temporal-duration-metadata.tsv',
                       metadata=True)
    df = convert_to_tsv(dev_location+temporal_metadata_file, dev_location + 'dev-temporal-relation-metadata.tsv',
                       metadata=True)

    df = convert_to_tsv(dev_location+"recast_red_metadata.json", dev_location + 'dev-red-metadata.tsv', metadata=True)
    df = convert_to_tsv(dev_location+"recast_tbdense_metadata.json", dev_location + 'dev-tbdense-metadata.tsv', metadata=True)
    df = convert_to_tsv(dev_location+"recast_tempeval3_metadata.json", dev_location + 'dev-tempeval3-metadata.tsv', metadata=True)
        
    ## Test
    df = convert_to_tsv(test_location+duration_metadata_file, test_location + 'test-temporal-duration-metadata.tsv',
                       metadata=True)
    df = convert_to_tsv(test_location+temporal_metadata_file, test_location + 'test-temporal-relation-metadata.tsv',
                       metadata=True)

    df = convert_to_tsv(test_location+"recast_red_metadata.json", test_location + 'test-red-metadata.tsv', metadata=True)
    df = convert_to_tsv(test_location+"recast_tbdense_metadata.json", test_location + 'test-tbdense-metadata.tsv', metadata=True)
    df = convert_to_tsv(test_location+"recast_tempeval3_metadata.json", test_location + 'test-tempeval3-metadata.tsv', metadata=True)
        
    ## Train
    df1 = convert_to_tsv(train_location+dur_metadata_file_split1, None, save=False,metadata=True)
    df2 = convert_to_tsv(train_location+dur_metadata_file_split2, None,save=False,metadata=True)
    df3 = convert_to_tsv(train_location+dur_metadata_file_split3, None,save=False, metadata=True)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['index'] = df.index
    df.to_csv(train_location + 'train-temporal-duration-metadata.tsv', sep='\t', index=False)
    print(f"File saved as: {train_location + 'train-temporal-duration-metadata.tsv'}")
    
    
    df1 = convert_to_tsv(train_location+rel_metadata_file_split1, None, save=False, metadata=True)
    df2 = convert_to_tsv(train_location+rel_metadata_file_split2, None, save=False, metadata=True)
    df3 = convert_to_tsv(train_location+rel_metadata_file_split3, None, save=False, metadata=True)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['index'] = df.index
    df.to_csv(train_location + 'train-temporal-relation-metadata.tsv', sep='\t', index=False)
    print(f"File saved as: {train_location + 'train-temporal-relation-metadata.tsv'}")
    
    df = convert_to_tsv(train_location+"recast_red_metadata.json", train_location + 'train-red-metadata.tsv', metadata=True)
    df = convert_to_tsv(train_location+"recast_tbdense_metadata.json", train_location + 'train-tbdense-metadata.tsv', metadata=True)
    df = convert_to_tsv(train_location+"recast_tempeval3_metadata.json", train_location + 'train-tempeval3-metadata.tsv', metadata=True)
    
if __name__ == '__main__':
    main()

