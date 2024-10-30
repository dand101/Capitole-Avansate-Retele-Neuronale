# config/optimizers.py
import torch.optim as optim


def get_optimizer(model, config):
    optimizer_type = config['optimizer']['type']
    learning_rate = config['optimizer']['learning_rate']
    weight_decay = config['optimizer'].get('weight_decay', 0.0)

    if optimizer_type == "SGD":
        momentum = config['optimizer'].get('momentum', 0.0)
        nesterov = config['optimizer'].get('nesterov', False)

        if nesterov and not momentum:
            raise ValueError("Nesterov requires momentum for SGD.")
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                         nesterov=nesterov)

    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "RmsProp":
        momentum = config['optimizer'].get('momentum', 0.0)
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    else:
        raise ValueError(f"Optimizer '{optimizer_type}' is not supported.")
