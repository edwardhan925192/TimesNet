import pickle
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from tqdm import tqdm
from timesmodel import Model
from whole_dataset import TimeSeriesDataset,TimeSeries_ValDataset,TimeSeries_TestDataset
import json
import pandas as pd
import copy
#from times_config import configs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, output_type, df_train, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion):
    '''
    1. Takes model trainset and validation set
    2. Load Data with datasets
    3. Train (schedular, criterion, optimizer)
    4. Predict validation set
    '''

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    model_type = model

    # ==================== MODEL SELECTION ========================== #
    if model == 'timesnet':
      model = Model(configs).to(device)

    # ==================== CRITERION ========================== #
    if criterion =='mse':
        criterion = nn.MSELoss()
    if criterion =='mae':
        criterion = nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0005)

    # ==================== TRAINING ========================== #
    if configs.task_name == 'short_term_forecast':        
        train_dataset = TimeSeriesDataset(output_type, df_train, configs.seq_len, configs.pred_len, target_col)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)

                if outputs.shape[-1] <= 1:
                  outputs = outputs.squeeze(-1)
                
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                scheduler.step(epoch + batch_idx / len(train_loader))

            average_training_loss = total_loss / len(train_loader)

            # Validation step after each epoch for each validation set
            model.eval()
            total_val_loss = 0

            for val_df in df_validation:
                val_dataset = TimeSeries_ValDataset(output_type, val_df, configs.seq_len, configs.pred_len, target_col, batch_sizes)
                val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False)

                val_loss = 0
                with torch.no_grad():
                    for batch_data, batch_target in val_loader:
                        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                        outputs = model(batch_data)

                        if outputs.shape[-1] <= 1:
                          outputs = outputs.squeeze(-1)
                        # ========= OUTPUT ADJUSTED ======= # 
                        outputs_adjusted = outputs[:, :batch_target.size(1)]
                        loss = criterion(outputs_adjusted, batch_target)
                        # ================================= # 
                        
                        val_loss += loss.item()

                total_val_loss += val_loss / len(val_loader)

            average_validation_loss = total_val_loss / len(df_validation)

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_training_loss}, Average Validation Loss: {average_validation_loss}")

            # Update best validation loss and epoch
            if average_validation_loss < best_val_loss:
                best_val_loss = average_validation_loss
                best_epoch = epoch
                print(outputs)
                best_model_state = copy.deepcopy(model.state_dict())


    print(f"Best Epoch: {best_epoch + 1} with Validation Loss: {best_val_loss}")

    return best_epoch, best_model_state

def test_model(model, output_type, test,target_col,learning_rate, num_epochs,batch_sizes, configs, criterion):
    '''
    Retrain the model with full datasets and make a final prediction
    '''

    model_type = model
    best_model_state = None
    # ==================== MODEL SELECTION ========================== #
    if model == 'timesnet':
      model = Model(configs).to(device)

    # ==================== CRITERION ========================== #
    if criterion =='mse':
        criterion = nn.MSELoss()
    if criterion =='mae':
        criterion = nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0005)

    test_dataset = TimeSeries_TestDataset(test, configs.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)
    
    train_dataset = TimeSeriesDataset(output_type, test, configs.seq_len, configs.pred_len, target_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)

                optimizer.zero_grad()
                outputs = model(batch_data)

                if outputs.shape[-1] <= 1:
                  outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_loader))

    best_model_state = copy.deepcopy(model.state_dict())
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_test_data in test_loader:
            batch_test_data = batch_test_data.to(device)
            print(batch_test_data)
            outputs = model(batch_test_data)
            predictions.extend(outputs.cpu().numpy())

    return predictions,best_model_state

# ====================== Train, Test MAIN ========================= #
def timesnetmain(model,output_type,df_train, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion):
    if model == 'timesnet':
      model = Model(configs).to(device)

    train_data = train
    # ===== train and validate model ===== #
    best_epoch, train_model_state = train_model(model,output_type, df_train, df_validation,  target_col, learning_rate, num_epochs, batch_sizes, configs, criterion)

    # from validation get best epoch and retrain with full datasets and return the prediction of last one
    best_epoch = best_epoch + 1
    full_training_set = df_test

    # ===== test model ===== #
    pred, test_model_state = test_model(model, output_type, df_test,target_col, learning_rate, best_epoch, batch_sizes,configs, criterion)
    return pred,train_model_state, test_model_state

def test_model_with_weights(model_type, state_dict_path, test,  configs):
    '''
    Retrain the model with full datasets and make a final prediction
    '''      

    model = Model(configs).to(device)
    # Assuming state_dict is an OrderedDict containing model weights
    model.load_state_dict(state_dict_path)        
    model.eval()  # Set the model to evaluation mode

    # Prepare the test data
    test_dataset = TimeSeries_TestDataset(test, configs.seq_len)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_test_data in test_loader:
            batch_test_data = batch_test_data.to(device)
            outputs = model(batch_test_data)
            predictions.extend(outputs.cpu().numpy())

    return predictions

def timesnetmodel_experiment(model,output_type,df_train, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion):
    if model == 'timesnet':
      model = Model(configs).to(device)
    
    # ===== train and validate model ===== #
    best_epoch, train_model_state = train_model(model,output_type, df_train, df_validation,  target_col, learning_rate, num_epochs, batch_sizes, configs, criterion)

    # from validation get best epoch and retrain with full datasets and return the prediction of last one
    best_epoch = best_epoch + 1    

    # ===== using weight that are gained from train ===== #    
    pred = test_model_with_weights(None, train_model_state, df_test, batch_sizes, configs, criterion)

    return pred,train_model_state,best_epoch
