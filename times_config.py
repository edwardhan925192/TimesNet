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
        self.top_k = 4        # FFT frequency
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

configs = Config()

# pred 180 got 2.4 currently
