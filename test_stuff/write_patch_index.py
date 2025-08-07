from pathlib import Path
import torch
from tqdm import tqdm

from multiprocessing import Pool
import pickle
import pandas as pd
import time
#main dict - key:tomo folder path, value:patch_dict
#patch_dict- key:patch filename, value: has_motor:bool

#get master folder path
#for each tomo,




def write_per_tomo_dict(tomo_folder:Path):
    local_patch_dict = {}
    patch_filepaths = [x for x in tomo_folder.iterdir() if x.is_file()]
    # print(patch_filepaths)
    for patch_filepath in patch_filepaths:
        patch_dict = torch.load(str(patch_filepath))
        labels = patch_dict['labels']
        truth_value = bool(labels[0, 3])#shape is only 1,3 for patch labels for now
        #labels[0,3] should only be 0 or 1 so yea
        patch_filename = patch_filepath.name
        local_patch_dict[patch_filename] = truth_value
        
    return local_patch_dict

if __name__ == '__main__':
    tomo_dict = {}
    folder_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\relabel_data')
    tomo_folders = [x for x in folder_path.iterdir() if x.is_dir()]
    
    with Pool(processes=12) as pool:
        results = pool.map(write_per_tomo_dict, tomo_folders)
        
    for i, patch_dict in enumerate(results):
        tomo_dict[str(tomo_folders[i].name)] = patch_dict
    
    pickle.dump(tomo_dict, open(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\test_stuff/tomo_dict.pkl', 'wb'))
    
    
    tomo_dict = pickle.load(open(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\test_stuff/tomo_dict.pkl', 'rb'))
    print(len(tomo_dict))
    
    
    table = []
    for tomo_id in tqdm(tomo_dict,ncols=160,desc = 'Writing to CSV'):
        
        patch_dict = tomo_dict[tomo_id]
        
        for patch_id in patch_dict:
            #tomo_id, patch_id, has_motor
            has_motor = patch_dict[patch_id]
            row = [tomo_id, patch_id, has_motor]
            table.append(row)
        
    df = pd.DataFrame(data = table, columns=['tomo_id', 'patch_id', 'has_motor'])    
    df.to_csv(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_relabel_index.csv', index = False)

    # df.add()
# if __name__ == '__main__':
#    items = list(range(int(1e8)))
#    with Pool(processes=8) as pool:
#        results = pool.map(worker_function, items)
#    print(results)
    #write motor bools
    