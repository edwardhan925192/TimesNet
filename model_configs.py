import torch

'''
self.seq_len = selecting range of sequences that user want to evaluate

# SEQ_LEN IS CHOSEN INSIDE THE RANGE OF PRED_LEN

self.eval_range = selecting features that user want to evaluate
'''


class SchedulerConfig:
    def __init__(self):
        # STARTS from lr goes down to eta_min in T_0 
        self.CosineAnnealingWarmRestarts = {'T_0': 10, 'T_mult': 1, 'eta_min': 0.0005}
        self.StepLR = {'step_size': 10, 'gamma': 0.1}
        self.ExponentialLR = {'gamma': 0.95}

        #steps_per_epoch should be set to number of batches , epochs should be total number of epochs 
        # Starts from low lr to max lr 
        self.OneCycleLR = {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20}        

        self.CyclicLR = {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5,'step_size_down':5,  'mode': 'triangular'}        

    def get_params(self, scheduler_name):
        return getattr(self, scheduler_name, None)
        
num_features = 3
target_name = '평균기온'

class Config:
    def __init__(self):
        # ==================== Task name ================ #
        self.task_name = 'short_term_forecast'
        # Output features and c_out should be the same when the task is anomaly_detection

        # =================== Datasets Shape================== #
        self.seq_len = 365
        self.window_shift = 1
        self.enc_in = num_features    # Features

        # ================== MODEL ====================== #
        self.d_model = 20     # Convolution Embedding dimension AFTER RESHAPING
        self.top_k = 4        # FFT frequency
        self.d_ff = 20       # Convolution Output layer dimension AFTER RESHAPING
        self.num_kernels = 6  # inception block Num of different grid cells used / If using dcvn set it to 3
        self.dropout = 0.2130496718140465    # Dropout rate
        self.e_layers = 1     # num Timeblock
        self.label_len = num_features   # Features

        self.target_col = target_name   # Name of target column
        self.cnn_type = 'inceptionv1' # dcvn (KERNEL = 3), inceptionv1, inceptionv2, res_dcvn, res_inceptionv1, res_inceptionv2

        # ================= Output shape ================= #
        self.pred_len = 358   # Prediction length
        self.c_out = 1        # Output feature
        self.eval_range = 0
        self.seq_range = np.concatenate([np.arange(0, 45), np.arange(317, 358)]) #(1 less)

        # ================= Scheduler Configurations ========= #
        self.scheduler_config = SchedulerConfig()
        self.scheduler_name = 'CosineAnnealingWarmRestarts' #'CosineAnnealingWarmRestarts', 'StepLR', 'ExponentialLR', 'OneCycleLR', 'CyclicLR'
        self.scheduler_update_type = 'epoch' # epoch, batch

    def update(self, new_params):
        for key, value in new_params.items():
            if hasattr(self, key):
                setattr(self, key, value)       

configs = Config()



# ========================== Itransformer configuration ================================= # 
# ========================== Itransformer configuration ================================= # 
# ========================== Itransformer configuration ================================= # 
# ========================== Itransformer configuration ================================= # 
# ========================== Itransformer configuration ================================= # 
num_features = 5
target_name = ''

class SchedulerConfig:
    def __init__(self):
        # STARTS from lr goes down to eta_min in T_0
        self.CosineAnnealingWarmRestarts = {'T_0': 20, 'T_mult': 1, 'eta_min': 0.0001}
        self.StepLR = {'step_size': 10, 'gamma': 0.1}
        self.ExponentialLR = {'gamma': 0.95}

        #steps_per_epoch should be set to number of batches , epochs should be total number of epochs
        # Starts from low lr to max lr
        self.OneCycleLR = {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20}

        self.CyclicLR = {'base_lr': 0.0001, 'max_lr': 0.01, 'step_size_up': 3,'step_size_down':3,  'mode': 'triangular'}

    def get_params(self, scheduler_name):
        return getattr(self, scheduler_name, None)

class Itransformer_Config:
    def __init__(self):
        # ==================== Task name ================ #
        self.task_name = 'short_term_forecast'
        # Output features and c_out should be the same when the task is anomaly_detection

        # =================== Datasets Shape================== #
        self.seq_len = 365
        self.enc_in = num_features    # Features

        # ================== MODEL ====================== #
        self.d_model = 22     # Embedding dimension
        self.d_ff = 22      # Output layer dimension
        self.dropout = 0.1    # Dropout rate
        self.e_layers = 3     # Transformer block

        self.target_col = target_name   # location of target column

        self.embed = 'fixed' # embedding type
        self.freq = 'h'      # embedding frequency
        self.output_attention = None
        self.factor = 5
        self.n_heads = 8
        self.activation = None

        # ================= Output shape ================= #
        self.pred_len = 358
        self.c_out = 1        # Output feature
        self.eval_range = 0
        self.seq_range = np.concatenate([np.arange(0, 45), np.arange(317, 358)]) #(1 less)

        # ================= Extras =================== #
        self.use_norm = True
        self.class_strategy = None

        # ================= Scheduler Configurations ========= #
        self.scheduler_config = SchedulerConfig()
        self.scheduler_name = 'CosineAnnealingWarmRestarts' #'CosineAnnealingWarmRestarts', 'StepLR', 'ExponentialLR', 'OneCycleLR', 'CyclicLR'
        self.scheduler_update_type = 'epoch' # epoch, batch

    def update(self, new_params):
        for key, value in new_params.items():
            if hasattr(self, key):
                setattr(self, key, value)       

configs = Itransformer_Config()
