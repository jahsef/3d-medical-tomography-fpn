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

    def __init__(self, model,train_loader, val_loader, optimizer, scheduler,
                 regression_loss_fn, conf_loss_fn, regression_loss_weight, conf_loss_weight,device,
                 save_dir='./models/'):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.regression_loss_fn = regression_loss_fn
        self.conf_loss_fn = conf_loss_fn
        self.regression_loss_weight = regression_loss_weight
        self.conf_loss_weight = conf_loss_weight
        
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        

        # Send model to device once
        self.model.to(self.device)

    def _train_one_epoch(self, epoch_index):
        regression_loss_tracker = utils.LossTracker(is_mean_loss=True)
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)

        self.model.train()
        total_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                                 desc=f"Epoch {epoch_index + 1}")
        progress_bar.ncols = 100

        for batch_idx, (inputs, labels) in progress_bar:
            batch_size = inputs.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type= "cuda"):
                outputs = self.model(inputs)
                #outputs (b,max_motors,4)
                regression_loss = self.regression_loss_fn(outputs[..., :3], labels[..., :3])
                conf_loss = self.conf_loss_fn(outputs[..., 3], labels[..., 3])
                
            weighted_regression_loss = self.regression_loss_weight * regression_loss
            weighted_conf_loss = self.conf_loss_weight * conf_loss
            combined_loss = weighted_regression_loss + weighted_conf_loss
            
            combined_loss.backward()
            
            self.optimizer.step()

            regression_loss_tracker.update(batch_loss=regression_loss.item(), batch_size=batch_size)
            conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=batch_size)

            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == total_batches:
                regression_loss = regression_loss_tracker.get_epoch_loss()
                conf_loss = conf_loss_tracker.get_epoch_loss()
                progress_bar.set_postfix(loss=f"regression loss: {regression_loss:.4f}, conf loss: {conf_loss:.4f}")

        return regression_loss_tracker.get_epoch_loss(), conf_loss_tracker.get_epoch_loss(), combined_loss


    def _validate_one_epoch(self):
        regression_loss_tracker = utils.LossTracker(is_mean_loss=True)
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)
        
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                batch_size = inputs.shape[0]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type= "cuda"):
                    outputs = self.model(inputs)
                    #outputs (b,max_motors,4)
                    regression_loss = self.regression_loss_fn(outputs[..., :3], labels[..., :3])
                    conf_loss = self.conf_loss_fn(outputs[..., 3], labels[..., 3])
                    
                weighted_regression_loss = self.regression_loss_weight * regression_loss
                weighted_conf_loss = self.conf_loss_weight * conf_loss
                combined_loss = weighted_regression_loss + weighted_conf_loss
                regression_loss_tracker.update(batch_loss=regression_loss.item(), batch_size=batch_size)
                conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=batch_size)
                

        return regression_loss_tracker.get_epoch_loss(), conf_loss_tracker.get_epoch_loss(), combined_loss

    def train(self, epochs=50, save_period=0):
        csv_filepath = os.path.join(self.save_dir, "train_results.csv")
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=self.save_dir)
        
        best_val_total_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, 'best.pt')
        
        for epoch in range(epochs):
            train_regression_loss, train_conf_loss, train_total_loss = self._train_one_epoch(epoch)
            val_regression_loss, val_conf_loss, val_total_loss = self._validate_one_epoch()
            self.scheduler.step()
            
            print(f"Epoch {epoch}:")
            print(f"  Train Regression Loss: {train_regression_loss:.6f} | Val Regression Loss: {val_regression_loss:.6f}")
            print(f"  Train Conf Loss      : {train_conf_loss:.6f} | Val Conf Loss      : {val_conf_loss:.6f}")
            print(f"  Train Total Loss     : {train_total_loss:.6f} | Val Total Loss     : {val_total_loss:.6f}")
            print("-" * 60)  # separator line for readability


            log_metrics(epoch,
                train_regression_loss=train_regression_loss,
                val_regression_loss=val_regression_loss,
                train_conf_loss=train_conf_loss,
                val_conf_loss=val_conf_loss,
                train_total_loss=train_total_loss,
                val_total_loss=val_total_loss,
                filename=csv_filepath
            )

            if val_total_loss < best_val_total_loss:
                best_val_total_loss = val_total_loss
                
                logging.info(f"Validation accuracy improved. Saving model to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)

            if save_period != 0 and (epoch + 1) % save_period == 0:
                periodic_save_path = os.path.join(self.save_dir, f"epoch{epoch+1}.pt")
                torch.save(self.model.state_dict(), periodic_save_path)

        logger.graph_training(csv_path=csv_filepath)

if __name__ == '__main__':
    fart = Trainer()
    fart.logger.graph_training(csv_path= r'C:\Users\kevin\Documents\GitHub\MNIST_pytorch\models\cifar10\skinny_deep_res\basic+rand_augment50epoch\train_results.csv')
    plt.show()