import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, output_type, dataframe, sequence_length, prediction_length, target_column):
        """
        Takes single data frame and return sequence length and targets(single)

        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        prediction_length (int): The length of the prediction sequences.
        target_column (str): The name of the target column to predict.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.output_type = output_type

        # Convert DataFrame to a PyTorch tensor
        self.data_tensor = torch.tensor(self.dataframe.values).float()

        # Get the index of the target column
        self.target_idx = self.dataframe.columns.get_loc(target_column)

    def __len__(self):
        """
        Return the total number of samples available in the dataset.
        """
        return len(self.dataframe) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        start_idx = index
        end_idx = start_idx + self.sequence_length

        # Input features (all columns)
        input_sequence = self.data_tensor[start_idx:end_idx, :]

        if self.output_type == 'single':
          # Target output (only the target column)
          target_sequence = self.data_tensor[end_idx:end_idx + self.prediction_length, self.target_idx]

        if self.output_type == 'whole':
          target_sequence = self.data_tensor[end_idx:end_idx + self.prediction_length,:]

        return input_sequence, target_sequence

class TimeSeries_ValDataset(Dataset):
    def __init__(self,output_type, dataframe, sequence_length, prediction_length, target_column, batch_size):
        """
        RETURN THE FIRST SEQUENCE OF DATA

        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        prediction_length (int): The length of the prediction sequences.
        target_column (str): The name of the target column to predict.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.batch_size = batch_size

        self.output_type = output_type
        # Convert DataFrame to a PyTorch tensor
        self.data_tensor = torch.tensor(self.dataframe.values).float()

        # Get the index of the target column
        self.target_idx = self.dataframe.columns.get_loc(target_column)

    def __len__(self):
        """
        Return the total number of samples available in the dataset.
        """
        #return self.batch_size
        return 1


    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        start_idx = index
        end_idx = start_idx + self.sequence_length

        # Input features (all columns)
        input_sequence = self.data_tensor[start_idx:end_idx, :]

        if self.output_type == 'single':
          # Target output (only the target column)
          target_sequence = self.data_tensor[end_idx:end_idx + self.prediction_length, self.target_idx]

        if self.output_type == 'whole':
          target_sequence = self.data_tensor[end_idx:end_idx + self.prediction_length,:]
        
        return input_sequence, target_sequence

class TimeSeries_TestDataset(Dataset):
    def __init__(self, dataframe, sequence_length):
        """
        RETURN THE LAST SEQUENCE OF DATA

        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        batch_size (int): The number of sequences to return from the end of the DataFrame.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length        

        # Convert DataFrame to a PyTorch tensor
        self.data_tensor = torch.tensor(self.dataframe.values).float()

    def __len__(self):
        """
        Return the total number of samples available in the dataset.
        """
        # return self.batch_size
        return 1

    def __getitem__(self, index):
        """
        Generate one sample of data for testing.
        """
        # Calculate the start index for each sequence
        total_length = len(self.dataframe)
        start_idx = total_length - self.sequence_length + index
        end_idx = start_idx + self.sequence_length

        # Ensure the index is within the bounds of the dataframe
        if start_idx < 0 or end_idx > total_length:
            raise IndexError("Index out of bounds for sequence generation")

        # Input features (all columns)
        input_sequence = self.data_tensor[start_idx:end_idx, :]

        return input_sequence
