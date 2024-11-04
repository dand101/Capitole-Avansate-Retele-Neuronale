# models/lenet.py
import torch.nn as nn
import torch
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self._initialize_fc1(input_size)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _initialize_fc1(self, input_size):
        dummy_input = torch.zeros(1, 1, input_size, input_size)
        x = F.max_pool2d(F.relu(self.conv1(dummy_input)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        self.fc1 = nn.Linear(x.numel(), 120)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_lenet_model(num_classes):
    return LeNet(num_classes=num_classes)
