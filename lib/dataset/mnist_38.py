#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from lib.dataset.randaugment import RandAugmentMC

mnist_mean = (0.1307,)
mnist_std = (0.3081,)


def random_subset(dataset, n_samples, seed):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    return Subset(dataset, indices=indices[:n_samples])


class MNIST38(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = torch.tensor(self.targets)
        idx = (self.targets == 3) | (self.targets == 8)
        self.targets = self.targets[idx]
        self.targets[self.targets == 3] = 0
        self.targets[self.targets == 8] = 1
        self.data = self.data[idx]


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


TRANSFORM_LABELED_MNIST = transforms.Compose(
    [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std),
    ]
)

TRANSFORM_UNLABELED_MNIST = TransformFixMatch(mean=mnist_mean, std=mnist_std)
TRANSFORM_TEST_MNIST = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mnist_mean, std=mnist_std)]
)


def get_mnist_38(root: str, student_dataset_max_size: int, student_seed: int):
    labeled_dataset = MNIST38(
        root=root, train=True, download=True, transform=TRANSFORM_LABELED_MNIST
    )
    test_dataset = MNIST38(
        root=root, train=False, download=True, transform=TRANSFORM_TEST_MNIST
    )
    unlabeled_dataset = MNIST38(
        root=root, train=True, download=True, transform=TRANSFORM_UNLABELED_MNIST
    )
    student_dataset = random_subset(
        dataset=MNIST38(
            root=root, train=True, download=True, transform=TRANSFORM_LABELED_MNIST
        ),
        n_samples=student_dataset_max_size,
        seed=student_seed,
    )

    return {
        "labeled": labeled_dataset,
        "unlabeled": unlabeled_dataset,
        "test": test_dataset,
        "student": student_dataset,
    }
