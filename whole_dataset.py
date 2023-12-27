import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, prediction_length, seq_range, eval_range):
        """
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        prediction_length (int): The length of the prediction sequences.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.seq_range = seq_range
        self.eval_range = eval_range
        self.data_tensor = torch.tensor(self.dataframe.values).float()

    def __len__(self):
        return len(self.dataframe) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.sequence_length
        target_end_idx = end_idx + self.prediction_length

        input_sequence = self.data_tensor[start_idx:end_idx, :]
        target_sequence = self.data_tensor[end_idx:target_end_idx, :]
        target_sequence_ = target_sequence[:, self.eval_range] if self.seq_range is None else target_sequence[self.seq_range, self.eval_range]    

        return input_sequence, target_sequence_

class TimeSeries_ValDataset(Dataset):
    def __init__(self, dataframe, sequence_length, prediction_length, seq_range, eval_range):
        """
        TAKES DATAFRAME AND RETURN A SINGLE SEQUENCE TOKEN AND TARGET TOKEN
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        prediction_length (int): The length of the prediction sequences.        
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.seq_range = seq_range
        self.eval_range = eval_range
        self.data_tensor = torch.tensor(self.dataframe.values).float()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.sequence_length
        target_end_idx = end_idx + self.prediction_length

        input_sequence = self.data_tensor[start_idx:end_idx, :]
        target_sequence = self.data_tensor[end_idx:target_end_idx, :]
        target_sequence_ = target_sequence[:, self.eval_range] if self.seq_range is None else target_sequence[self.seq_range, self.eval_range]    

        return input_sequence, target_sequence_

class TimeSeries_TestDataset(Dataset):
    def __init__(self, dataframe, sequence_length):
        """
        TAKES A DF AND RETURN A SINGLE SEQUENCE TOKEN 
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        sequence_length (int): The length of the input sequences.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length        
        self.data_tensor = torch.tensor(self.dataframe.values).float()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        total_length = len(self.dataframe)
        start_idx = total_length - self.sequence_length + index
        end_idx = start_idx + self.sequence_length

        if start_idx < 0 or end_idx > total_length:
            raise IndexError("Index out of bounds for sequence generation")

        input_sequence = self.data_tensor[start_idx:end_idx, :]
        return input_sequence
