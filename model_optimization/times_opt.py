import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from times_traintest import train_model

def timesnet_opt(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool,trials):
  def objective(trial):
      # ============ Params to OPT ============== # 
      batch_sizes = trial.suggest_int('batch_sizes', 30, 64)
      d_ff = trial.suggest_int('half_d_ff', 9, 12) * 2
      top_k = trial.suggest_int('top_k', 3, 5)
      drop_out = trial.suggest_float('dropout', 0.05, 0.25)
      
      configs.d_ff = d_ff
      configs.d_model = d_ff
      configs.top_k = top_k
      configs.drop_out = drop_out

      # ============ Train for bayseian opt ============== #
      training_loss_history, validation_loss_history, mean_validation_loss_per_epoch, best_epoch, best_model_state = train_model(
          model_type=model,
          df_train=df_train_,
          df_validation=df_validation,
          target_col=target_col,
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          batch_sizes=batch_sizes,
          configs=configs,
          criterion=criterion,  # or 'mae' based on your requirement
          schedular_bool=scheduler_bool
      )

      # ============ Minimum val ============== #
      return min(mean_validation_loss_per_epoch)

  # Run the Optuna study
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=trials)

  # Get the best hyperparameters and the best score
  best_hyperparams = study.best_params
  best_score = study.best_value
  print('Best hyperparameters:', best_hyperparams)
  print('Best score:', best_score)

  return best_hyperparams, best_score
