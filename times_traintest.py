import pickle
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from times_model import Model
from itransformer import Itransformer
from whole_dataset import TimeSeriesDataset,TimeSeries_ValDataset,TimeSeries_TestDataset
from schedular.scheduler import initialize_scheduler
import json
import pandas as pd
import copy
import torchvision.ops
#from times_config import configs
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pickle
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from times_model import Model
from itransformer import Itransformer
from schedular.scheduler import initialize_scheduler
import json
import pandas as pd
import copy
import torchvision.ops
#from times_config import configs
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model_type, df_train, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, schedular_bool):
    '''
    Both training set and validation set have to be LISTS. 
    If you want specific target for outputs specific target_col (single name) 

    1. Takes model trainset and validation set
    2. Load Data with datasets
    3. Train (schedular, criterion, optimizer)
    4. Predict validation set
    '''

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1    

    training_loss_history = []
    validation_loss_history = []

    # ==================== TARGET INDEX ========================== #
    col_list = list(df_train[0].columns)
    target_index = col_list.index(target_col) if target_col in col_list else -1
    
    # ==================== CRITERION ========================== #
    if criterion =='mse':
        criterion = nn.MSELoss()
    if criterion =='mae':
        criterion = nn.L1Loss()    

    # ==================== TRAINING ========================== #
    whole_val_storage = []

    # ==================== INDIVIDUAL TRAINING SETS ====================== # 
    for train_ , val_ in zip(df_train, df_validation):                
      val_storage = []
        
      # ==================== MODEL SELECTION ========================== #
      if model_type == 'timesnet':
          model = Model_output(configs).to(device)
      if model_type == 'itransformer':
          model = Itransformer(configs).to(device)

      # ==================== OPTIM ========================= # 
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      # =================== Schedular initialization ======================= #
      if schedular_bool:
        scheduler = initialize_scheduler(optimizer, configs)
          
      if configs.task_name == 'short_term_forecast':
          train_dataset = TimeSeriesDataset(train_, configs.seq_len, configs.pred_len, configs.seq_range, configs.eval_range)
          train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)

      for epoch in range(num_epochs):          

          model.train()
          total_loss = 0

          # ================== INDIVIDUAL EPOCH ===================== #
          for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
              batch_data, batch_target = batch_data.to(device), batch_target.to(device)
              optimizer.zero_grad()
              outputs = model(batch_data)                            

              # ============== LOSSES ================ #
              loss = criterion(outputs, batch_target)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()

              # ========== Schedular ============= #
              if schedular_bool and configs.scheduler_update_type == 'batch':
                scheduler.step(epoch + batch_idx / len(train_loader))

          # Update Scheduler after each epoch if specified
          if schedular_bool and configs.scheduler_update_type == 'epoch':
              scheduler.step()

          average_training_loss = total_loss / len(train_loader)

          # =================== Validation ====================== #                                   
          model.eval()
          total_val_loss = 0

          for val_df in df_validation:
              val_dataset = TimeSeries_ValDataset(val_, configs.seq_len, configs.pred_len, configs.seq_range, configs.eval_range)
              val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False)

              val_loss = 0

              with torch.no_grad():
                  for batch_data, batch_target in val_loader:
                      batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                      outputs = model(batch_data)                                    
                      
                      # ============== LOSSES =============== # 
                      loss_ = criterion(outputs, batch_target)

                      val_loss += loss_.item()

              total_val_loss += val_loss / len(val_loader)                                            

          average_validation_loss = total_val_loss / len(df_validation)
          training_loss_history.append(average_training_loss)
          validation_loss_history.append(average_validation_loss)
          print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_training_loss}, Average Validation Loss: {average_validation_loss}")

          val_storage.append(average_validation_loss)              

          # Update best validation loss and epoch
          if average_validation_loss < best_val_loss:
              best_val_loss = average_validation_loss
              best_epoch = epoch
              print('=================Current_BEST======================')
              best_model_state = copy.deepcopy(model.state_dict())

      whole_val_storage.append(val_storage)
    
    
    mean_validation_loss_per_epoch = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*whole_val_storage)]
    best_epoch = -1 
    temp_loss = 99999999999
    for i in range(len(mean_validation_loss_per_epoch)):
      print(f"Final BEST Epoch {i + 1}/{num_epochs}, Final Average Validation Loss: {mean_validation_loss_per_epoch[i]}")    
      if mean_validation_loss_per_epoch[i] < temp_loss: 
          temp_loss = mean_validation_loss_per_epoch[i]
          best_epoch = i + 1 
        
    return training_loss_history, validation_loss_history, mean_validation_loss_per_epoch, best_epoch, best_model_state 

def test_model(model_type, df_test, target_col,learning_rate, num_epochs,batch_sizes, configs, criterion, scheduler_bool):
    '''
    Retrain the model with full datasets and make a final prediction and return model state 
    '''
    # ==================== TARGET INDEX ========================== #
    col_list = list(df_test.columns)
    target_index = col_list.index(target_col) if target_col in col_list else -1
    
    best_model_state = None
    # ==================== MODEL SELECTION ========================== #
    if model_type == 'timesnet':
      model = Model(configs).to(device)
    if model_type == 'itransformer':
      model = Itransformer(configs).to(device)

    # ==================== CRITERION ========================== #
    if criterion =='mse':
        criterion = nn.MSELoss()
    if criterion =='mae':
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # =================== Scheduler initialization ======================= # 
    if scheduler_bool:      
      scheduler = initialize_scheduler(optimizer, configs)    
    
    test_dataset = TimeSeries_TestDataset(df_test, configs.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)

    train_dataset = TimeSeriesDataset(df_test, configs.seq_len, configs.pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=False)

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)

                optimizer.zero_grad()
                outputs = model(batch_data)

                # ============== OUTPUT ADJUSTMENT =============== #
                if target_col:
                    outputs = outputs[:,:, target_index]
                    batch_target = batch_target[:,:, target_index]

                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()

                # ========== SCHEDULER ========== #                
                if scheduler_bool and configs.scheduler_update_type == 'batch':
                  scheduler.step(epoch + batch_idx / len(train_loader))

            # Update Scheduler after each epoch if specified
        if scheduler_bool and configs.scheduler_update_type == 'epoch':
          scheduler.step()

    best_model_state = copy.deepcopy(model.state_dict())

    # ====================== Testing ========================= #
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_test_data in test_loader:
            batch_test_data = batch_test_data.to(device)
            outputs = model(batch_test_data)

            # ============== OUTPUT ADJUSTMENT =============== #
            if target_col:
                  outputs = outputs[:,:, target_col]
                  batch_target = batch_target[:,:, target_col]

            predictions.extend(outputs.cpu().numpy())

    return predictions,best_model_state
