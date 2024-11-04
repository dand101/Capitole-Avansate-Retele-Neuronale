# models/resnet.py
from timm import create_model
import torch.nn as nn


def get_resnet18_model(num_classes, pretrained=True):
    model = create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
    model = nn.Sequential(
        nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        model
    )

    return model
