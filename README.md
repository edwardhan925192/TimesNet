# TimesNet 
![Overall purpose](https://github.com/edwardhan925192/Project3/assets/127165920/d1fb6548-e819-4ece-ba9e-e3922bba8c3e)  
The model meticulously employs the Fast Fourier Transform (FFT) to ascertain the predominant frequencies, thereby facilitating a deeper understanding of the underlying patterns. To further enhance this process, the model has incorporated the sophisticated Inception block for feature extraction. This advanced mechanism ensures a comprehensive capture of both interperiod and intraperiod variations of fluctuations. By doing so, the model promises a holistic and nuanced understanding of the data, transcending traditional analytical methodologies.


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
```
# References 
- Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long. **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**. In: International Conference on Learning Representations (ICLR), 2023.  


