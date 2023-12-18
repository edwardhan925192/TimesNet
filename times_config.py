import torch

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
        self.d_model = 20     # Embedding dimension
        self.top_k = 3        # FFT frequency
        self.d_ff = 20       # Output layer dimension
        self.num_kernels = 6  # inception block에서 / If using dcvn set it to 3
        self.dropout = 0.1    # Dropout rate
        self.e_layers = 1     # num Timeblock
        self.label_len = num_features   # Features

        self.target_col = target_name   # Name of target column
        self.cnn_type = 'inceptionv1' # dcvn, inceptionv1, inceptionv2, res_dcvn, res_inceptionv1, res_inceptionv2

        # ================= Output shape ================= #
        self.pred_len = 358   # Prediction length
        self.c_out = 1        # Output feature

        # ================= Scheduler Configurations ========= #
        self.scheduler_config = SchedulerConfig()
        self.schedular_name = 'CosineAnnealingWarmRestarts' #'CosineAnnealingWarmRestarts', 'StepLR', 'ExponentialLR', 'OneCycleLR', 'CyclicLR'
        self.schedular_type = 'batch' # epoch, batch 

configs = Config()
