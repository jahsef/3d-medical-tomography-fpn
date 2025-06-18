import torch
import time
import csv
import logging
from tqdm import tqdm  # Optional: For progress bars
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import utils
import gc
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    
def log_metrics(epoch, filename="logs.csv", **kwargs):
    fieldnames = list(kwargs.keys())
    values = [round(float(v), 6) for v in kwargs.values()]

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['epoch'] + fieldnames)  # Add epoch as first column

        writer.writerow([epoch] + values)
        
class Trainer:

    def __init__(self, model, batches_per_step,train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn,device,
                save_dir='./models/'):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        # self.regression_loss_fn = regression_loss_fn
        self.conf_loss_fn = conf_loss_fn
        # self.regression_loss_weight = regression_loss_weight
        # self.conf_loss_weight = conf_loss_weight
        
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir

        # Send model to device once
        self.model.to(self.device)
        
    def _compute_batch_loss(self, patches, labels):
        """
        Compute loss for batch duh
        
        Args:
            patches (torch.Tensor): Input patches tensor.
            xyzconf (torch.Tensor): Ground truth xyz + confidence values.
            valid_mask (torch.Tensor): precomputed bool mask if conf > 0, used for regression loss masking
            
        Returns:
            conf_loss (torch.Tensor)
        """


        
        patches = patches.to(self.device)
        # assert labels.shape == (1,4), f'labels shape wrong?: {labels.shape}'
        #i forgot labels are batched here lol
        # valid_mask = (labels[... , 3] > 0).bool()  # Shape: [b,max_motors]
        
        labels = labels.to(self.device)
        # valid_mask = valid_mask.to(self.device)
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = self.model(patches)

            #outputs, labels are b,1,d,h,w now
            #can just squeeze it down
            
            conf_loss = self.conf_loss_fn(outputs, labels)
            
            # bce_loss = F.binary_cross_entropy_with_logits(outputs, labels)
            # print(f"Focal: {conf_loss:.4f}, BCE: {bce_loss:.4f}")


        return conf_loss 
    
    def _train_one_epoch(self, epoch_index):
        # Clear CUDA cache at start of epoch
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)

        self.model.train()
        total_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                            desc=f"Epoch {epoch_index}")
        progress_bar.ncols = 100

        for batch_idx, (patches, labels, global_coords) in progress_bar:
            patches: torch.Tensor
            labels: torch.Tensor
            
            if batch_idx % self.batches_per_step == 0:#0,5,10
                self.optimizer.zero_grad()

            conf_loss = self._compute_batch_loss(patches, labels)
            
            # combined_loss.backward()
            conf_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 100)
            
            if (batch_idx + 1) % self.batches_per_step == 0:#4,9
                self.optimizer.step()
                    
            self.scheduler.step()

            conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])


            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == total_batches:
                progress_bar.set_postfix(
                    loss= f"conf loss: {conf_loss_tracker.get_epoch_loss():.4f}"
                )
                
            # # Clear memory at end of batch
            # if (batch_idx + 1) % 10 == 0:  # Every 10 batches
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            #     gc.collect()
        

        conf_loss = conf_loss_tracker.get_epoch_loss()
        
        return conf_loss


    def _validate_one_epoch(self):
        #THIS IS FOR TRAINING SCRIPT LATER
        # 1. create a new dir for full tomograms in .pt format (only from my validation stuff since it takes 10000000 gb of data)
        # 2. i have paths to my tomogram patch directories right, so at start of runtime i would reconstruct the full tomogram dirs from my list (i dont want to manually do this)
        # 3. pass in the list of paths to validation
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)

        self.model.eval()

        with torch.no_grad():
            for patches, labels, global_coords in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                conf_loss= self._compute_batch_loss(patches, labels)
                conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])
        conf_loss = conf_loss_tracker.get_epoch_loss()
        return conf_loss

    def train(self, epochs=50, save_period=0):
        csv_filepath = os.path.join(self.save_dir, "train_results.csv")
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=self.save_dir)
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, 'best.pt')
        optimizer_path = os.path.join(self.save_dir, 'best_optimizer.pt')
        
        for epoch in range(epochs):
            train_conf_loss = self._train_one_epoch(epoch)
            # val_regression_loss, val_conf_loss, val_total_loss = self._validate_one_epoch()
            # val_conf_loss = 0
            val_conf_loss = self._validate_one_epoch()
            
            print(f"Epoch {epoch}:")
            print(f"  Train Conf Loss      : {train_conf_loss:.6f} | Val Conf Loss      : {val_conf_loss:.6f}")
            print("-" * 60)  # separator line for readability


            log_metrics(epoch,
                train_conf_loss=train_conf_loss,
                val_conf_loss=val_conf_loss,
                filename=csv_filepath
            )

            if val_conf_loss < best_val_loss or val_conf_loss == 0:

                best_val_loss = val_conf_loss
                
                logging.info(f"Validation accuracy improved. Saving model to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(self.optimizer.state_dict(), optimizer_path)

            if save_period != 0 and (epoch + 1) % save_period == 0:
                periodic_save_path = os.path.join(self.save_dir, f"epoch{epoch}.pt")
                periodic_optimizer_save_path = os.path.join(self.save_dir, f"epoch{epoch}_optimizer.pt")
                torch.save(self.model.state_dict(), periodic_save_path)
                torch.save(self.optimizer.state_dict(), periodic_optimizer_save_path)
                
        logger.graph_training(csv_path=csv_filepath)

if __name__ == '__main__':
    fart = Trainer()
    fart.logger.graph_training(csv_path= r'C:\Users\kevin\Documents\GitHub\MNIST_pytorch\models\cifar10\skinny_deep_res\basic+rand_augment50epoch\train_results.csv')
    plt.show()