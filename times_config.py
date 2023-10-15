class Config:
    def __init__(self):
        # ==================== Task name ================ # 
        self.task_name = 'anomaly_detection'

        # Output features and c_out should be the same when the task is anomaly_detection

        # =================== Datasets Shape================== # 
        self.seq_len = 24    
        self.window_shift = 1        
        self.enc_in = 4    # Features          

        # ================== MODEL ====================== #  
        self.d_model = 32     # Embedding dimension
        self.top_k = 5        # FFT frequency
        self.d_ff = 32        # Output layer dimension
        self.num_kernels = 6  # inception block에서 사용될 커널숫자 
        self.dropout = 0    
        self.e_layers = 1     # num Timeblock 
        self.label_len = 24  # 예측할 sequence                 

        # ================= Output shape ================= # 
        self.pred_len = 24  
        self.c_out = 4        # Output feature
        
        # ================ Training ================ # 
        self.val == False
        self.lr = 0.001
        self.epochs = 1 
        self.batch_sizes = 24 

configs = Config()
