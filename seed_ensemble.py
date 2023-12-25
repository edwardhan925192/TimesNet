import numpy as np
from times_traintest import train_model
from times_maincode import test_model_with_weights, timesnetmain

def seed_ensemble_with_weights(model_type, df_train_, df_validation, df_test,  target_col, configs, learning_rate, num_epochs, batch_sizes, num_seed):

  model_states_lists = []

  # ================== TRAINING MODELS WITH DIFFERENT SEEDS ===================== #
  for seed in range(0,num_seed):
    seed_everything(seed)
    _,_,_,best_epoch,train_model_state = train_model(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool)
    model_states_lists.append(train_model_state)

  # =================== AFTER GETTING THE WEIGHTS PREDICT THE DATASETS WITH GAINED WEIGHTS ============ #
  prediction_lists = []

  for model_state in model_states_lists:
    pred = test_model_with_weights(model_type, model_state, df_test, target_col,  batch_sizes, configs)
    prediction_lists.append(pred)

  ensembled_pred = np.mean(np.stack(prediction_lists), axis=0)

  return prediction_lists, model_states_lists

def seed_ensemble(model_type,df_train, df_validation, df_test,  target_col, configs, learning_rate, num_epochs, batch_sizes, num_seed):

  model_states_lists = []
  prediction_lists = []

  # ================== TRAINING MODELS WITH DIFFERENT SEEDS ===================== #
  for seed in range(0,num_seed):
    seed_everything(seed)
    pred, model_state = timesnetmain(model,df_train, df_validation, df_test, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool)

  return prediction_lists,model_states_lists
