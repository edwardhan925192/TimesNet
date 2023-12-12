import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, prediction_length, target_column):
        """
        Initialize the dataset with a pandas DataFrame.

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
