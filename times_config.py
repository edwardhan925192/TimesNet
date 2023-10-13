class Config:
    def __init__(self, train_dataset, val_dataset, test_dataset):
        # ==================== Task name ================ # 
        self.task_name = 'anomaly_detection'

        # Output features and c_out should be the same when the task is anomaly_detection

        # =================== DATASETS ================== # 
        self.seq_len = 24    #예측에 사용할 sequence 
        self.window_shift = 1        
        self.enc_in = 4       #예측 feature

        # ================== MODEL ====================== #  
        self.d_model = 32     # Embedding dimension
        self.top_k = 5        # FFT frequency
        self.d_ff = 32        # Output layer dimension
        self.num_kernels = 6  # inception block에서 사용될 커널숫자 
        self.dropout = 0    
        self.e_layers = 1     # Timeblock 층 수 
        self.label_len = 24  # 예측할 sequence 
        self.pred_len = 24   # 예측할 sequence
        self.c_out = 4        # Output feature

        # ================ Data instances ================ # 
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset

        # ================ Training ================ # 
        self.lr = 0.001
        self.epochs = 1 
        self.batch_sizes = 24 

configs = Config(train,val,test)
