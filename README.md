# Project goal
**Main goal: Predicting movement of double pendulum using TimesNet model.**  
The dynamics of a double pendulum, inherently chaotic in nature, have historically posed challenges for precise analytical prediction using conventional mathematical algorithms. To address this complex system's unpredictability, we turned to a state-of-the-art deep learning model, 'TimesNet', designed specifically for time series forecasting and intricate pattern recognition. By employing 'TimesNet', our goal was to unravel the intricate dynamics and thereby accurately predict the motion of the double pendulum. This innovative approach aims to bridge the gap between traditional physics-based models and the potential of modern machine learning techniques.

# TimesNet 
![Overall purpose](https://github.com/edwardhan925192/Project3/assets/127165920/d1fb6548-e819-4ece-ba9e-e3922bba8c3e)  
The model meticulously employs the Fast Fourier Transform (FFT) to ascertain the predominant frequencies, thereby facilitating a deeper understanding of the underlying patterns. To further enhance this process, the model has incorporated the sophisticated Inception block for feature extraction. This advanced mechanism ensures a comprehensive capture of both interperiod and intraperiod variations of fluctuations. By doing so, the model promises a holistic and nuanced understanding of the data, transcending traditional analytical methodologies.
# Double pendulum
https://github.com/edwardhan925192/Project3/assets/127165920/3e701dc2-b622-4221-8b5a-2c5cf2501359



# TimesNet Usage
python 'times_traintest.py' --train_path 'path.csv' --val_path 'path.csv' --test_path 'path.csv' --lr 0.001 --epochs 1 --batch_sizes 2

# References 
- Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long. **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**. In: International Conference on Learning Representations (ICLR), 2023.  


