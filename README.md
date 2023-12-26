# TimesNet 
![Overall purpose](https://github.com/edwardhan925192/Project3/assets/127165920/d1fb6548-e819-4ece-ba9e-e3922bba8c3e)  
The model meticulously employs the Fast Fourier Transform (FFT) to ascertain the predominant frequencies, thereby facilitating a deeper understanding of the underlying patterns. To further enhance this process, the model has incorporated the sophisticated Inception block for feature extraction. This advanced mechanism ensures a comprehensive capture of both interperiod and intraperiod variations of fluctuations. By doing so, the model promises a holistic and nuanced understanding of the data, transcending traditional analytical methodologies.


# Seed
```markdown
import random
import numpy as np
import os
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_number = 42
seed_everything(seed_number)
```

# 0. Colab setup
```markdown
!pip install einops
!pip install optuna
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
!rm -r foldername
```

# 1. Setting up Validationsets
```markdown
!git clone 'https://github.com/edwardhan925192/TimesNet.git'
%cd '/content/TimesNet'

# Both train and test datasets takes dataframe
# train datasets returns single target and n sequences
# test datasets returns last of n batch of sequences without targets
# validation range = whats predicted
# seq_length = data used for prediction 

from validation_split import split_train_validation_timeseries
```
# 2. Optimizing and predicting
```markdown
seed_everything() #Currently set it to 42 
# ======== UPDATING configs ============ #
configs.update(best_param)
configs__ = configs

# ======== Datasets ========= # 
df_train_ = train_dfs # Must be list
df_validation = val_dfs # Must be list
df_test = test_dfs # This has to be full datasets

# ======== Parameters ========= # 
target_col = None # None or single string 
learning_rate = 0.001
num_epochs = 20
batch_sizes = 30
model = 'itransformer'
criterion = 'mae'
scheduler_bool = True 
trials = 20

# TRAIN
_,_,_,best_epoch,train_model_state = train_model(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool)

# TEST
pred, model_state = test_model(model, df_test, target_col,learning_rate, best_epoch,batch_sizes, configs, criterion, scheduler_bool)

# TRAIN AND TEST 
pred, state = timesnetmain(model,df_train_, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, schedular_bool)

# OPTIMIZATION 
best_param, best_score =  timesnet_opt(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool,trials)
```

# Ensemble 
```markdown
pred, modelstate = seed_ensemble(model_type,df_train, df_validation, df_test,  target_col, configs, learning_rate, num_epochs, batch_sizes, criterion, scheduler_bool, num_seed)
```
# RUNNING with weights and configs 
```markdown
# =========== SAVING =========== #
# Saving configs
with open('configs13.pkl', 'wb') as f:
    pickle.dump(configs, f)

# Specify a path for saving the model state
save_path = 'timesnet13.pth'

# Save the best model state
torch.save(train_model_state, save_path)

# =========== LOADING =========== #
from times_maincode import timesnetmodel_experiment, train_model,test_model,test_model_with_weights

config_path = '/content/TimesNet/timesnet109.pkl'
model_path = '/content/TimesNet/timesnet109.pth'

model_used = 'timesnet'

with open(config_path, 'rb') as file:
    config_data = pickle.load(file)

model_dict = torch.load(model_path)

pred = test_model_with_weights(model_used,'single',model_dict,train_ ,'평균기온',1, config_data )
```

# References 
- Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long. **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**. In: International Conference on Learning Representations (ICLR), 2023.  


