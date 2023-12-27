# model optimization
```markdown
from model_optimization.times_opt import timesnet_opt
from model_optimization.times_opt import itransformer_opt

# target_col should be set to None
best_params, best_score = timesnet_opt(model, df_train, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool,trials)
best_params, best_score = itransformer_opt(model, df_train_, df_validation, target_col, learning_rate, num_epochs, batch_sizes, configs, criterion, scheduler_bool,trials)
```
