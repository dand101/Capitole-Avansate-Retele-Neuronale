# config/schedulers.py
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def get_scheduler(optimizer, config):
    scheduler_type = config['scheduler']['type']

    if scheduler_type == "StepLR":
        step_size = config['scheduler'].get('step_size', 10)
        gamma = config['scheduler'].get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "ReduceLROnPlateau":
        patience = config['scheduler'].get('patience', 5)
        factor = config['scheduler'].get('gamma', 0.1)
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)

    elif scheduler_type == "None":
        return None

    else:
        raise ValueError(f"Scheduler '{scheduler_type}' is not supported.")
