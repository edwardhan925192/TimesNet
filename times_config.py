# config.py
class Config:
    def __init__(self):
        self.task_name = "short_term_forecast" #'anomaly_detection'
        # =================== DATASETS ================== # 
        self.seq_len = 168    #예측에 사용할 sequence 
        self.window_shift = 1        
        self.enc_in = 4       #예측 feature

        # ================== MODEL ====================== #  
        self.d_model = 32     #Embedding dimension
        self.top_k = 5        #FFT frequency
        self.d_ff = 32        #Output layer dimension
        self.num_kernels = 6  #inception block에서 사용될 커널숫자 
        self.dropout = 0    
        self.e_layers = 1     #Timeblock 층 수 
        self.label_len = 168  #예측할 sequence 
        self.pred_len = 168   #예측할 sequence
        self.c_out = 4        #Output feature

configs = Config()
