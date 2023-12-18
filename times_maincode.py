import pandas as pd 
import numpy as np 
from times_traintest import train_model, test_model
from times_model import Model
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================== Train, Test MAIN ========================= #
def timesnetmain(model,output_type,df_train, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, schedular_bool):
    if model == 'timesnet':
      model = Model(configs).to(device)

    train_data = train
    # ===== train and validate model ===== #
    _,_,best_epoch, train_model_state = train_model(model,output_type, df_train, df_validation,  target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, schedular_bool)

    # from validation get best epoch and retrain with full datasets and return the prediction of last one
    best_epoch = best_epoch + 1
    full_training_set = df_test

    # ===== test model ===== #
    pred, test_model_state = test_model(model, output_type, df_test,target_col, learning_rate, best_epoch, batch_sizes,configs, criterion, schedular_bool)
    return pred,train_model_state, test_model_state

def test_model_with_weights(model_type, output_type, state_dict_path, test, target_col,  batch_sizes, configs):
    '''
    Retrain the model with full datasets and make a final prediction
    '''
    # ==================== TARGET INDEX ========================== #
    col_list = list(test.columns)
    target_index = col_list.index(target_col) if target_col in col_list else -1

    model = Model(configs).to(device)
    # Assuming state_dict is an OrderedDict containing model weights
    model.load_state_dict(state_dict_path)

    model.eval()  # Set the model to evaluation mode

    # Prepare the test data
    test_dataset = TimeSeries_TestDataset(test, configs.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_test_data in test_loader:
            batch_test_data = batch_test_data.to(device)
            outputs = model(batch_test_data)

            # ============= OUTPUT TYPE AND SHAPE ================= #
            if output_type == 'single':
              outputs = outputs[:, :, target_index]

            if outputs.shape[-1] <= 1:
              outputs = outputs.squeeze(-1)
            predictions.extend(outputs.cpu().numpy())

    return predictions

def timesnetmodel_experiment(model,output_type,df_train, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, range_exp):
    if model == 'timesnet':
      model = Model(configs).to(device)

    train_data = df_train
    # ===== train and validate model ===== #
    _, _, best_epoch, train_model_state = train_model(model,output_type, df_train, df_validation,  target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, range_exp)

    # from validation get best epoch and retrain with full datasets and return the prediction of last one
    best_epoch = best_epoch + 1

    # ===== using weight that are gained from train ===== #
    pred = test_model_with_weights(None, output_type, train_model_state, df_test, target_col, batch_sizes, configs)

    return pred,train_model_state,best_epoch

def train_with_lr_range(model, output_type, df_train, df_validation, target_col, lr_range, num_epochs, batch_sizes, configs, criterion):
    '''
    Trains the model over a range of learning rates and plots the training process.

    Parameters:
    - model: The model to train.
    - output_type: The output type of the model.
    - df_train: The training dataset.
    - df_validation: The validation dataset.
    - target_col: The target column in the datasets.
    - lr_range: A tuple indicating the start and end of the learning rate range (start_lr, end_lr).
    - num_epochs: Number of epochs to train for each learning rate.
    - batch_sizes: The batch size for training.
    - configs: Configuration settings for the model.
    - criterion: The loss function.
    - bool_schedular: A flag to enable/disable learning rate experiments.
    '''
    start_lr, end_lr = lr_range
    num_steps = 5  # Number of steps between start_lr and end_lr
    learning_rates = np.linspace(start_lr, end_lr, num_steps)

    for lr in learning_rates:
        print(f"Training with Learning Rate: {lr}")
        training_loss_history, validation_loss_history, _, _ = train_model(model, output_type, df_train, df_validation, target_col, lr, num_epochs, batch_sizes, configs, criterion, None)

        # Plot the loss history for each learning rate
        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, training_loss_history, label=f'Training Loss - LR: {lr}')
        plt.plot(epochs, validation_loss_history, label=f'Validation Loss - LR: {lr}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epochs at LR: {lr}')
        plt.legend()
        plt.show()
