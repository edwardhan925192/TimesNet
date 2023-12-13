import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, prediction_length, target_column):
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

        # Target output (only the target column)
        target_sequence = self.data_tensor[end_idx:end_idx + self.prediction_length, self.target_idx]

        return input_sequence, target_sequence

class TimeSeries_ValTestDataset(Dataset):
    def __init__(self, dataframe, validation_df, target_column, sequence_length, prediction_length, is_test):
        """
        Initialize the dataset with a pandas DataFrame for testing and validation.

        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        validation_df (pd.DataFrame): The validation DataFrame.
        target_column (str): The name of the target column in the validation DataFrame.
        sequence_length (int): The length of the input sequences.
        prediction_length (int): The length of the target sequences.
        is_test (bool): Flag to indicate if the dataset is for testing (True) or validation (False).
        """
        self.dataframe = dataframe
        self.validation_df = validation_df
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.is_test = is_test
        self.data_tensor = torch.tensor(self.dataframe.values).float()
        if not is_test:
            self.target_tensor = torch.tensor(self.validation_df[self.target_column].values).float()

    def __len__(self):
        """
        Return the total number of samples available in the dataset.
        """
        return 1

    def __getitem__(self, index):
        """
        Generate one sample of data for testing or validation.
        """
        # For testing and validation, return the last 'sequence_length' observations
        start_idx = len(self.dataframe) - self.sequence_length
        input_sequence = self.data_tensor[start_idx:, :]

        if self.is_test:
            return input_sequence
        else:
            # For validation, return the target values for the prediction length
            target_value = self.target_tensor[:self.prediction_length]
            return input_sequence, target_value
