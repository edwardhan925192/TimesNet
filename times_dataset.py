import torch 
from torch.utils.data import Dataset, DataLoader

class TimesNetDataset(Dataset):
    def __init__(self, data_array, configs, train=True):
        self.data_array = torch.from_numpy(data_array).float()
        self.sequence_length = configs.seq_len
        self.prediction_length = configs.pred_len
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.data_array) - self.sequence_length - self.prediction_length + 1
        else:
            return (len(self.data_array)- self.prediction_length) // (self.sequence_length)


    def __getitem__(self, index):
        if self.train:
            return (
                self.data_array[index:index+self.sequence_length],
                self.data_array[index+self.sequence_length:index+self.sequence_length+self.prediction_length]
            )
        else:
            # For test datasets, each chunk starts at index * (sequence_length + prediction_length) to avoid overlap
            start_idx = index * (self.sequence_length)
            input_data = self.data_array[start_idx:start_idx+self.sequence_length]
            target_data = self.data_array[start_idx+self.sequence_length:start_idx+self.sequence_length+self.prediction_length]
            return input_data, target_data

# ==================== Anomaly detection ==================== #
class TimesNetAnomalyDataset(Dataset):
    def __init__(self, data_array, configs, train=True, window_shift=1):
        self.data_array = torch.from_numpy(data_array).float()
        self.sequence_length = configs.seq_len        
        self.train = train
        self.window_shift = configs.window_shift

    def __len__(self):        
        return (len(self.data_array)) // (self.sequence_length + self.window_shift)

    def __getitem__(self, index):                
        start_idx = index * self.window_shift
        return self.data_array[start_idx:start_idx+self.sequence_length]
               


