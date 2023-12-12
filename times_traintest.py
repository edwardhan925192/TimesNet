import pickle
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from timesmodel import Model
from whole_dataset import TimesNetDataset,TimesNetAnomalyDataset
import json
import pandas as pd
#from times_config import configs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train, target_col, learning_rate, num_epochs, batch_sizes, configs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # ========================== DATASETS (Task_type) ========================== # 
    if configs.task_name == 'short_term_forecast':
        train_dataset = TimesNetDataset(train, configs, target_col, train=True)    
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)        

    if configs.val: 
        val_dataset = TimesNetDataset_valtest(train, configs, target_col, train=True)    
        val_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)        
        
    elif configs.task_name == 'anomaly_detection':
        train_dataset = TimesNetAnomalyDataset(np.array(train), configs)    
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)
    # =========================================================================== #

    # ========================== TRAIN (Task_type) ========================== # 
    if configs.task_name == 'short_term_forecast':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_data, batch_target in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
            # Validation step             
            if configs.val:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data, batch_target in val_loader:
                        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_target)
                        val_loss += loss.item()
    
                print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}")

            scheduler.step()
            
    elif configs.task_name == 'anomaly_detection':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        scheduler.step()
    # =========================================================================== #

def test_model(model, test,batch_sizes, configs):
    criterion = nn.MSELoss()
    test_dataset = TimesNetDataset(np.array(test), configs, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)

    model.eval()
    test_loss = 0
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            test_loss += loss.item()            
            
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_target.cpu().numpy())

    print(f"Testing Loss: {test_loss / len(test_loader)}")
    return predictions, true_values

def save_results(predictions, true_values, filename='results.csv'):
    # Convert predictions and true_values to DataFrame
    pred_df = pd.concat([pd.DataFrame(item) for item in predictions], ignore_index=True)
    true_df = pd.concat([pd.DataFrame(item) for item in true_values], ignore_index=True)

    combined_df = pd.concat([pred_df, true_df], axis=1)
    combined_df.columns = [f"Pred_{i}" for i in range(pred_df.shape[1])] + [f"True_{i}" for i in range(true_df.shape[1])]

    combined_df.to_csv(filename, index=False)

def load_data_from_path(filepath):
    """Load data from the given path into a pandas dataframe."""
    return pd.read_csv(filepath)
    
# ================================================================= #
# ====================== Train, Test MAIN ========================= #
# ================================================================= #

model = Model(configs).to(device)

train_data = train

if configs.val == True:
    val_data = val
else:
    val_data = None

train_model(model, train_data, configs.lr, configs.epochs, configs.batch_sizes, configs, val_data)

# Test the model on each test dataset
all_predicted_values = []
all_ground_truth_values = []

predicted_values, ground_truth_values = test_model(model, test_data, configs.batch_sizes, configs)

all_predicted_values.extend(predicted_values)
all_ground_truth_values.extend(ground_truth_values)
    
save_results(all_predicted_values, all_ground_truth_values)
