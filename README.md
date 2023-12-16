# TimesNet 
![Overall purpose](https://github.com/edwardhan925192/Project3/assets/127165920/d1fb6548-e819-4ece-ba9e-e3922bba8c3e)  
The model meticulously employs the Fast Fourier Transform (FFT) to ascertain the predominant frequencies, thereby facilitating a deeper understanding of the underlying patterns. To further enhance this process, the model has incorporated the sophisticated Inception block for feature extraction. This advanced mechanism ensures a comprehensive capture of both interperiod and intraperiod variations of fluctuations. By doing so, the model promises a holistic and nuanced understanding of the data, transcending traditional analytical methodologies.

# Colab
```markdown
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
```

# TimesNet Datasets
```markdown
!git clone 'https://github.com/edwardhan925192/TimsNet.git'

# Both train and test datasets takes dataframe
# train datasets returns single target and n sequences
# test datasets returns last of n batch of sequences without targets
```
# Training and testing TimesNet
```markdown
seed = 0
torch.manual_seed(seed)
# If you are using CUDA

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU.
np.random.seed(seed)
random.seed(seed)  # Replace 42 with your chosen seed number

df_train_ = train_ #df_train[-1]
df_validation = [val_f] # this should be a list
target_col = '평균기온'
learning_rate = 0.01
num_epochs = 20 
batch_sizes = 30
configs__ = configs
model = 'timesnet'
df_test = train_
output_type = 'single'

pred,train_model_state,best_epoch = timesnetmodel_experiment(model,output_type,df_train_, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs)
```
# RUNNING with weights and configs 
```markdown
from times_traintest import timesnetmodel_experiment, train_model,test_model,test_model_with_weights

config_path = '/content/configs13.pkl'
model_path = '/content/timesnet13.pth'

with open(config_path, 'rb') as file:
    config_data = pickle.load(file)

model_dict = torch.load(model_path)

pred = test_model_with_weights(None,model_dict,train_, config_data )
```

# References 
- Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long. **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**. In: International Conference on Learning Representations (ICLR), 2023.  


