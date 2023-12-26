import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from times_traintest import train_model

def itransformer_opt(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool,trials):
  def objective(trial):
      # Define the hyperparameters to be optimized by Optuna
      batch_sizes = trial.suggest_int('batch_sizes', 24,48)
      d_ff = trial.suggest_int('half_d_ff', 9, 12) * 2
      factor = trial.suggest_int('factor', 4, 6)
      n_heads = trial.suggest_int('factor', 7, 9)
      drop_out = trial.suggest_float('dropout', 0.05, 0.25)

      # Other hyperparameters can be defined similarly
      configs.d_ff = d_ff
      configs.d_model = d_ff
      configs.factor = factor
      configs.n_heads = n_heads
      configs.drop_out = drop_out

      # Call your training function with the suggested hyperparameters
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

      # Optuna tries to minimize the returned value, so return the metric you want to minimize
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
