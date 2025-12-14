import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import shutil
import inspect
import pandas as pd
import time



def create_file_list(dir:Path):
    """used to create a list of files from a dir, dont use on dir with subdirs"""
    poop_list = []
    dir = Path(dir)#ensure its path lol
    for path in dir.iterdir():
        if path.is_dir():
            raise Exception('theres a subdir in this dir, this function dont support')
        poop_list.append(path)
    return poop_list

# file_list = create_file_list(Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train'))
# print(file_list)
# time.sleep(20000)

def map_csvs_to_pt(csv_df:pd.DataFrame, list_tensor_paths:list[Path], max_motors:int) -> list[tuple]:
    #returns a list of tuple of tensor paths and the corresponding csv row
# For each sample
# {
#   "input": tensor of shape [C, H, W],  # your .pt tomograph
#   "coords": FloatTensor[20, 3],        # padded motor positions
#   "mask": BoolTensor[20],              # 1 if motor is real, 0 if padded
# }
    fart_list = []
    for path in list_tensor_paths:
        labels_dict = {}
        tomo_id = Path(path).stem#make sures its path
        indices = csv_df.index[csv_df['tomo_id'] == tomo_id]
        
        test_row = csv_df.loc[indices[0]]
        shape_series = test_row['Array shape (axis 0)':'Array shape (axis 2)']
        shape = []
        
        for axis, size in shape_series.items():
            #np.int64
            shape.append(int(size))
            
        shape = tuple(shape)
        
        coords = torch.zeros(size = (max_motors,3), dtype=torch.float32)
        mask = torch.zeros(size = (max_motors,), dtype=torch.float32)
        #assign coords and mask
        #we do a linear search on tomo mask, may benefit from converting to dict
        df_tomo_rows = csv_df.iloc[indices]
        df_coords = df_tomo_rows.loc[:, 'Motor axis 0' : 'Motor axis 2']
        np_coords = df_coords.to_numpy(dtype = np.float32)
        assert np_coords.shape[0] < max_motors, "number of rows in this data is greater than max motors, cant put labels into correct shape"
        num_rows = np_coords.shape[0]
        coords[0:num_rows, :] = torch.from_numpy(np_coords)
        # df_num_motors = df_tomo_rows.loc[:, 'Number of motors']
        # np_num_motors = df_num_motors.to_numpy(dtype =np.float32)
        mask[0:num_rows] = 1
        
        labels_dict['tomo_id'] = tomo_id
        labels_dict['shape'] = shape
        labels_dict['xyzconf'] = torch.cat([coords, mask.unsqueeze(-1)], dim = -1)
        
        #shape Array shape (axis 0), Array shape (axis 1), Array shape (axis 2)



        fart_list.append((path, labels_dict))

    return fart_list

class F1ScoreTracker:
    #f1 score = 2 * p * r / (p + r)
    #p = tp/ (tp + fp), r = tp / (tp + fn)
    def __init__(self, beta, tau):
        """beta is scaling value, tau is distance threshold in this case"""
        self.true_positives = 0
        self.false_negatives = 0
        self.tau = tau
        self.beta = beta
        
        pass
    def update(self, predictions, ground_truth):
        """_summary_

        Args:
            predictions (_type_): _description_
            ground_truth (_type_): (n,3) 
        """
        #have to write this lmao
        
        pass
        
  
    
class LossTracker:
    def __init__(self, is_mean_loss:bool):
        self.total_loss = 0
        self.total_samples = 0
        self.is_mean_loss = is_mean_loss

    def update(self, batch_loss, batch_size):
        
        if self.is_mean_loss:
            self.total_loss += batch_loss * batch_size 
        else:
            self.total_loss += batch_loss
  
            
        self.total_samples += batch_size

    def get_epoch_loss(self):
        return self.total_loss / self.total_samples if self.total_samples > 0 else 0
    
class AccuracyTracker:
    def __init__(self):
        self.incorrect = 0
        self.total_predictions = 0

    def update(self, predictions, ground_truth):
        #gets max across a row and preserves batch dim
        predicted_classes = torch.argmax(predictions, axis=1)  # Shape: (batch,)
        
        self.incorrect += torch.sum(predicted_classes != ground_truth)
        self.total_predictions += predictions.shape[0]

    def get_accuracy(self):
        correct = self.total_predictions - self.incorrect
        return (correct / self.total_predictions).item()



class Logger:
    def graph_training(self, csv_path):
        #read csv then plot 
        #should have accuracy and loss seperate
        file_exists = os.path.isfile(csv_path)
        if not file_exists:
            raise Exception('csv path dont exist lol')
        
        df = pd.read_csv(csv_path)
        
        epochs = df['epoch'].values
        train_loss, val_loss = df['train_loss'].values, df['val_loss'].values
        train_acc, val_acc = df['train_acc'].values, df['val_acc'].values
        
        parent_dir = Path(csv_path).parent
        
        self._create_plot(num = 0, title = 'Train/Val Loss vs Epoch', x_label= 'Epoch',y_label= 'Train/Val Loss')
        self._plot_helper(num = 0, label = 'Train Loss', x_data= epochs, y_data= train_loss, color= 'red')
        self._plot_helper(num = 0, label = 'Val Loss', x_data= epochs, y_data= val_loss, color= 'blue')
        plt.savefig(parent_dir / 'Loss.png')
        
        self._create_plot(num = 1, title = 'Train/Val Acc vs Epoch', x_label= 'Epoch',y_label= 'Train/Val Acc')
        self._plot_helper(num = 1, label = 'Train Acc', x_data= epochs, y_data= train_acc, color= 'red')
        self._plot_helper(num = 1, label = 'Val Acc', x_data= epochs, y_data= val_acc, color= 'blue')
        plt.savefig(parent_dir / 'Accuracy.png')
        
        #we can plot our loss vs epochs here
        
    def log_training_settings(self, save_dir):
        model_defs_dir = Path.cwd() / 'model_defs'
        if type(save_dir) == str:
            save_dir = Path(save_dir)
        #ensure its a path
        shutil.copytree(model_defs_dir, save_dir / 'model_defs', dirs_exist_ok=True)

        train_dir = Path.cwd() / 'train'
        shutil.copytree(train_dir, save_dir / 'train', dirs_exist_ok=True)

    def _create_plot(self,num,title, x_label, y_label):
        plt.figure(num = num, figsize=(10, 6))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
    def _plot_helper(self,num,label, x_data, y_data, color):   
        plt.figure(num = num)
        plt.plot(x_data, y_data,label=label, color = color)
        plt.legend()


        