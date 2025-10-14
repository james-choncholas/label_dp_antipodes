#!/usr/bin/env python3

import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)


def simple(num_classes):
    return SimpleModel(num_classes=num_classes)


class SimpleConvModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConvModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)


def simple_conv(num_classes):
    return SimpleConvModel(num_classes=num_classes)
