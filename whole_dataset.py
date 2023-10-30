import torch 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader

class TimesNetDataset(Dataset):
    def __init__(self, data_array, configs):
        if isinstance(data_array, pd.DataFrame):
            data_array = data_array.values
        self.data_array = torch.from_numpy(data_array).float()
        self.sequence_length = configs.seq_len
        self.prediction_length = configs.pred_len 
        self.window_shift = configs.window_shift if hasattr(configs, 'window_shift') else 1       

    def __len__(self):                
        total_shifted_length = len(self.data_array) - self.prediction_length - (self.sequence_length - 1) * self.window_shift        
        return (total_shifted_length - 1) // self.window_shift + 1
        


    def __getitem__(self, index):                    
        start_idx = index * self.window_shift
        return (
            self.data_array[start_idx:start_idx+self.sequence_length],
            self.data_array[start_idx+self.sequence_length:start_idx+self.sequence_length+self.prediction_length]
        )
        

# ==================== Anomaly detection ==================== #
class TimesNetAnomalyDataset(Dataset):
    def __init__(self, data_array, configs):
        self.data_array = torch.from_numpy(data_array).float()
        self.sequence_length = configs.seq_len                
        self.window_shift = configs.window_shift

    def __len__(self):        
        return ((len(self.data_array)- self.sequence_length) // (self.window_shift)) 

    def __getitem__(self, index):                
        start_idx = index * self.window_shift
        return self.data_array[start_idx:start_idx+self.sequence_length]
               


