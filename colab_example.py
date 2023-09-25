# ================== git clone ==================== # 
!git clone 'https://github.com/leekyuyoung20230313/1.team.git'


# ================== dir ================ # 
%cd '/content/1.team/TimsNet'

# ========== train, val, test dir path; lr, epochs, batch_sizes setups ======== #
## sample data used 
!python 'times_traintest.py' \
--train_path '/content/1.team/TimsNet/sample_data/s_train1.csv'\
'/content/1.team/TimsNet/sample_data/s_train2.csv'\ 
--val_path '/content/1.team/TimsNet/sample_data/s_val1.csv'\
'/content/1.team/TimsNet/sample_data/s_val2.csv'\ 
--test_path '/content/1.team/TimsNet/sample_data/s_test1.csv'\
'/content/1.team/TimsNet/sample_data/s_test2.csv'\ 
--lr 0.001\ 
--epochs 1\ 
--batch_sizes 2

# ============= Predicted data is saved in current dir as results.csv =========== #
import pandas as pd
result = pd.read_csv('/content/1.team/TimsNet/results.csv')
result
