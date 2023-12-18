import torch

def initialize_scheduler(optimizer, configs):

    scheduler_params = configs.scheduler_config.get_params(configs.scheduler_name)  # Get scheduler parameters

    if configs.scheduler_config == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif configs.scheduler_config == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif configs.scheduler_config == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif configs.scheduler_config == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    elif configs.scheduler_config == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
    else:
        raise ValueError("Unsupported scheduler type")

    return scheduler
    
