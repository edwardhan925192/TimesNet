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
from times_config import configs
from timesmodel import Model
from times_dataset import TimesNetDataset,TimesNetAnomalyDataset
import json
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train, validation, learning_rate, num_epochs, batch_sizes, configs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    
    # ========================== DATASETS (Task_type) ========================== # 
    if configs.task_name == 'short_term_forecast':
        train_dataset = TimesNetDataset(np.array(train), configs, train=True)    
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)
        val_dataset = TimesNetDataset(np.array(validation), configs, train=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False)
        
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
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_target)
                    val_loss += loss.item()
    
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test TimesNet model.')
    parser.add_argument('--train_paths', type=str, nargs='+', required=True, help='List of paths to training data files.')
    parser.add_argument('--val_paths', type=str, nargs='+', required=True, help='List of paths to validation data files.')
    parser.add_argument('--test_paths', type=str, nargs='+', required=True, help='List of paths to test data files.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training.')
    parser.add_argument('--batch_sizes', type=int, default=16, help='Number of batch sizes for training.')

    args = parser.parse_args()

    model = Model(configs).to(device)

    # Train and validate the model on each dataset in sequence
    for train_path, val_path in zip(args.train_paths, args.val_paths):
        train_data = load_data_from_path(train_path)
        val_data = load_data_from_path(val_path)

        train_model(model, train_data, val_data, args.lr, args.epochs, args.batch_sizes, configs)

    # Test the model on each test dataset
    all_predicted_values = []
    all_ground_truth_values = []

    for test_path in args.test_paths:
        test_data = load_data_from_path(test_path)
        predicted_values, ground_truth_values = test_model(model, test_data, args.batch_sizes, configs)

        all_predicted_values.extend(predicted_values)
        all_ground_truth_values.extend(ground_truth_values)
        

    save_results(all_predicted_values, all_ground_truth_values)
