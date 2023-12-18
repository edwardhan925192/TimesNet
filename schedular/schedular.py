import torch

def initialize_scheduler(optimizer, configs):

    scheduler_params = configs.schedular_config.get_params(configs.schedular_name)  # Get scheduler parameters

    if configs.schedular_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif configs.schedular_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif configs.schedular_name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif configs.schedular_name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    elif configs.schedular_name == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
    else:
        raise ValueError("Unsupported scheduler type")

    return scheduler
    
