# models/__init__.py
from .resnet18 import get_resnet18_model
from .mlp import get_mlp_model
from .lenet import get_lenet_model

from .preact_resnet18 import get_preact_resnet18_model


def get_model(model_config):
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    pretrained = model_config.get('pretrained', False)

    if model_name == 'resnet18':
        return get_resnet18_model(num_classes, pretrained)
    elif model_name == 'mlp':
        return get_mlp_model(num_classes)
    elif model_name == 'lenet':
        return get_lenet_model(num_classes)
    elif model_name == 'preact_resnet18':
        return get_preact_resnet18_model(num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
