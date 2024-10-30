# models/resnet.py
from timm import create_model
import torch.nn as nn


def get_resnet18_model(num_classes, pretrained=True):
    model = create_model("resnet18", pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
