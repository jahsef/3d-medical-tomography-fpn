import pandas as pd
import numpy as np

csv = pd.read_csv(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')


indices = csv.index[csv["tomo_id"] == "tomo_003acc"]
#we are just indexing by a boolean mask

rows = csv.iloc[indices]
print(rows)
print(type(rows))
#pandas df need row, column indexing
#series can just be column indexing

fart = rows.loc[:, 'Motor axis 0' : 'Motor axis 2']

np_arr = fart.to_numpy(dtype = np.float32)


print(fart)
print('lol')
print(np_arr)
