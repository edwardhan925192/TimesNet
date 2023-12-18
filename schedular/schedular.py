import torch

def initialize_scheduler(optimizer, configs):

    scheduler_params = configs.scheduler_config.get_params(configs.scheduler_name)  # Get scheduler parameters

    if configs.scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif configs.scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif configs.scheduler_name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif configs.scheduler_name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    elif configs.scheduler_name == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
    else:
        raise ValueError("Unsupported scheduler type")

    return scheduler

class SchedulerConfig:
    def __init__(self):
        self.CosineAnnealingWarmRestarts = {'T_0': 10, 'T_mult': 1, 'eta_min': 0.0005}
        self.StepLR = {'step_size': 10, 'gamma': 0.1}
        self.ExponentialLR = {'gamma': 0.95}
        self.OneCycleLR = {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20}
        self.CyclicLR = {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'mode': 'triangular'}
        # Additional schedulers can be added here

    def get_params(self, scheduler_name):
        return getattr(self, scheduler_name, None)
