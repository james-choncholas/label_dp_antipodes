#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Runs CIFAR10 and CIFAR100 training with ALIBI for Label Differential Privacy
"""
import argparse
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

from lib import models
from lib.alibi import Ohm, RandomizedLabelPrivacy, NoisedCIFAR
from lib.dataset.canary import fill_canaries
from lib.dataset.imdb import get_imdb_dataloaders
from lib.dataset.mnist_38 import get_mnist_38
from lib.models.sentiment import SentimentModel
from opacus.utils import stats
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from tqdm import tqdm

PAD_IDX = 1
#######################################################################
# Settings
#######################################################################
@dataclass
class LabelPrivacy:
    sigma: float = 0.1
    max_grad_norm: float = 1e10
    delta: float = 1e-5
    post_process: str = "mapwithprior"
    mechanism: str = "Laplace"
    noise_only_once: bool = True


@dataclass
class Learning:
    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    weight_decay: float = 1e-4
    random_aug: bool = False


@dataclass
class Settings:
    dataset: str = "cifar100"
    canary: int = 0
    arch: str = "wide-resnet"
    privacy: LabelPrivacy = field(default_factory=LabelPrivacy)
    learning: Learning = field(default_factory=Learning)
    gpu: int = -1
    world_size: int = 1
    out_dir_base: str = "/tmp/alibi/"
    data_dir_root: str = "/tmp/"
    seed: int = 0
    dataset_choices: list = field(default_factory=lambda: ["cifar10", "cifar100", "mnist", "mnist38", "imdb"])


MAX_GRAD_INF = 1e6

#######################################################################
# CIFAR transforms
#######################################################################
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


FIXMATCH_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
FIXMATCH_CIFAR10_STD = (0.2471, 0.2435, 0.2616)
FIXMATCH_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
FIXMATCH_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

#######################################################################
# Stat Collection settings
#######################################################################

# The following few lines, enable stats gathering about the run
_clipping_stats = {}  # will be used to collect stats from different layers
_norm_stats = {}  # will be used to find histograms


def enable_stats(stats_dir):

    if stats_dir is None:
        return None
    # 1. where the stats should be logged
    summary_writer = tensorboard.SummaryWriter(stats_dir)
    stats.set_global_summary_writer(summary_writer)
    # 2. enable stats
    stats.add(
        # stats on training accuracy
        stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=1),
        # stats on validation accuracy
        stats.Stat(stats.StatType.TEST, "accuracy"),
        stats.Stat(stats.StatType.TRAIN, "privacy", frequency=1),
    )
    return summary_writer


#######################################################################
# train, test, functions
#######################################################################
def save_checkpoint(state, filename=None):
    torch.save(state, filename)


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, criterion, device, dataset_name):
    model.train()
    losses = []
    acc = []

    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        if dataset_name == "imdb":
            text, text_lengths = batch.text
            targets = batch.label
            output = model(text, text_lengths)
            loss = criterion(output, targets)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = targets.detach().cpu().numpy()
        else:
            images = batch[0].to(device)
            targets = batch[1].to(device)
            labels = targets if len(batch) == 2 else batch[2].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, targets)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        acc.append(acc1)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

    return np.mean(acc), np.mean(losses)


def test(model, test_loader, criterion, epoch, device, dataset_name):
    model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if dataset_name == "imdb":
                text, text_lengths = batch.text
                targets = batch.label
                output = model(text, text_lengths)
                loss = criterion(output, targets)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = targets.detach().cpu().numpy()
            else:
                images = batch[0].to(device)
                target = batch[1].to(device)

                output = model(images)
                loss = criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
            
            acc1 = accuracy(preds, labels)
            losses.append(loss.item())
            acc.append(acc1)

    print(
        f"Test epoch {epoch}:",
        f"Loss: {np.mean(losses):.6f} ",
        f"Acc@1: {np.mean(acc) :.6f} ",
    )
    return np.mean(acc), np.mean(losses)


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 30:  # warm-up
        lr = lr * float(epoch + 1) / 30
    else:
        lr = lr * (0.2 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_model(arch: str, num_classes: int):
    if "wide" in arch.lower():
        print("Created Wide Resnet Model!")
        return models.wideresnet(
            depth=28,
            widen_factor=8 if num_classes == 100 else 4,
            dropout=0,
            num_classes=num_classes,
        )
    elif "simple" in arch.lower():
        print("Created Simple Model!")
        if "conv" in arch.lower():
            from lib.models.simple import simple_conv
            return simple_conv(num_classes=num_classes)
        from lib.models.simple import simple
        return simple(num_classes=num_classes)
    elif "sentiment" in arch.lower():
        print("Created Sentiment Model!")
        return SentimentModel(
            vocab_size=10000,
            embedding_dim=32,
            output_dim=num_classes,
            dropout=0.5,
            pad_idx=PAD_IDX,
        )
    else:
        print("Created simple Resnet Model!")
        return models.resnet18(num_classes=num_classes)


#######################################################################
# main worker
#######################################################################


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main_worker(settings: Settings):
    print(f"settings are {settings}")
    make_deterministic(settings.seed)
    out_dir_base = settings.out_dir_base
    os.makedirs(out_dir_base, exist_ok=True)

    best_acc = 0
    if settings.dataset.lower() == "cifar100":
        num_classes = 100
    elif settings.dataset.lower() == "cifar10":
        num_classes = 10
    elif settings.dataset.lower() == "mnist":
        num_classes = 10
    elif settings.dataset.lower() == "mnist38":
        num_classes = 2
    elif settings.dataset.lower() == "imdb":
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {settings.dataset}")

    model = create_model(settings.arch, num_classes)
    device = torch.device("cuda") if settings.gpu >= 0 else torch.device("cpu")
    model = model.to(device)

    # DEFINE LOSS FUNCTION (CRITERION)
    sigma = settings.privacy.sigma
    noise_only_once = settings.privacy.noise_only_once
    randomized_label_privacy = RandomizedLabelPrivacy(
        sigma=sigma,
        delta=settings.privacy.delta,
        mechanism=settings.privacy.mechanism,
        device=None if noise_only_once else device,
    )
    criterion = Ohm(
        privacy_engine=randomized_label_privacy,
        post_process=settings.privacy.post_process,
    )
    # DEFINE OPTIMIZER
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.learning.lr,
        momentum=settings.learning.momentum,
        weight_decay=settings.learning.weight_decay,
        nesterov=True,
    )
    # DEFINE DATA
    if settings.dataset.lower() == "imdb":
        imdb_data = get_imdb_dataloaders(
            root=settings.data_dir_root,
            batch_size=settings.learning.batch_size,
            device=device,
            student_seed=settings.seed
        )
        train_loader = imdb_data["labeled"]
        test_loader = imdb_data["test"]
        vocab = imdb_data["vocab"]
        # Overwrite PAD_IDX with the one from vocab
        global PAD_IDX
        PAD_IDX = imdb_data["pad_idx"]
        # Update model with the correct pad_idx
        model = create_model(settings.arch, num_classes)
        model = model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=settings.learning.lr,
            momentum=settings.learning.momentum,
            weight_decay=settings.learning.weight_decay,
            nesterov=True,
        )
        train_dataset = None # Not used in the same way for IMDB
    else:
        if settings.dataset.lower() == "mnist":
            CIFAR = MNIST
            rand_aug = []
            normalize = [
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        elif settings.dataset.lower() == "mnist38":
            CIFAR = get_mnist_38
            rand_aug = []
            normalize = [
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        else:
            CIFAR = CIFAR100 if settings.dataset.lower() == "cifar100" else CIFAR10
            rand_aug = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            ]
            if settings.dataset.lower() == "cifar100":
                normalize = [
                    transforms.ToTensor(),
                    transforms.Normalize(FIXMATCH_CIFAR100_MEAN, FIXMATCH_CIFAR100_STD),
                ]
            else:  # CIFAR-10
                normalize = [
                    transforms.ToTensor(),
                    transforms.Normalize(FIXMATCH_CIFAR10_MEAN, FIXMATCH_CIFAR10_STD),
                ]

        train_transform = transforms.Compose(
            rand_aug + normalize if settings.learning.random_aug else normalize
        )
        settings.data_dir_root = os.path.join(
            settings.data_dir_root, settings.dataset.lower()
        )
        if settings.dataset.lower() == "mnist38":
            datasets = get_mnist_38(
                root=settings.data_dir_root,
                student_dataset_max_size=1000,
                student_seed=settings.seed,
            )
            train_dataset = datasets["labeled"]
            test_dataset = datasets["test"]
        else:
            train_dataset = CIFAR(
                train=True,
                transform=train_transform,
                root=settings.data_dir_root,
                download=True,
            )
            test_dataset = CIFAR(
                train=False,
                transform=transforms.Compose(normalize),
                root=settings.data_dir_root,
                download=True,
            )
        if settings.canary > 0 and settings.canary < len(train_dataset):
            # capture debug info
            original_label_sum = sum(train_dataset.targets)
            original_last10_labels = [train_dataset[-i][1] for i in range(1, 11)]
            # inject canaries
            train_dataset = fill_canaries(
                train_dataset, num_classes, N=settings.canary, seed=settings.seed
            )
            # capture debug info
            canary_label_sum = sum(train_dataset.targets)
            canary_last10_labels = [train_dataset[-i][1] for i in range(1, 11)]
            # verify presence
            if original_label_sum == canary_label_sum:
                raise Exception(
                    "Canary infiltration has failed."
                    f"\nOriginal label sum: {original_label_sum} vs"
                    f" Canary label sum: {canary_label_sum}"
                    f"\nOriginal last 10 labels: {original_last10_labels} vs"
                    f" Canary last 10 labels: {canary_last10_labels}"
                )
        if noise_only_once:
            train_dataset = NoisedCIFAR(
                train_dataset, num_classes, randomized_label_privacy
            )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=settings.learning.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # test data
        test_loader = data.DataLoader(
            test_dataset, batch_size=settings.learning.batch_size, shuffle=False
        )

    cudnn.benchmark = True


    stats_dir = os.path.join(out_dir_base, "stats")
    summary_writer = enable_stats(stats_dir)

    for epoch in range(settings.learning.epochs):
        adjust_learning_rate(optimizer, epoch, settings.learning.lr)

        randomized_label_privacy.train()
        assert isinstance(criterion, Ohm)  # double check!
        if not noise_only_once:
            randomized_label_privacy.increase_budget()

        # train for one epoch
        model, train_loader, optimizer, criterion, device
        acc, loss = train(model, train_loader, optimizer, criterion, device, settings.dataset)

        epsilon, alpha = randomized_label_privacy.privacy
        label_change = 0
        if settings.dataset.lower() != "imdb":
            label_change = (
                train_dataset.label_change if noise_only_once else criterion.label_change
            )

        stats.update(
            stats.StatType.TRAIN,
            top1Acc=acc,
            loss=loss,
            epsilon=epsilon,
            alpha=alpha,
            label_change_prob=label_change,
        )

        # evaluate on validation set
        if randomized_label_privacy is not None:
            randomized_label_privacy.eval()
        acc, loss = test(model, test_loader, criterion, epoch, device, settings.dataset)
        stats.update(stats.StatType.TEST, top1Acc=acc, loss=loss)

        # remember best acc@1 and save checkpoint
        chkpt_file_name = os.path.join(out_dir_base, f"checkpoint-{epoch}.tar")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": settings.arch,
                "state_dict": model.state_dict(),
                "acc1": acc,
                "optimizer": optimizer.state_dict(),
            },
            chkpt_file_name,
        )
        if acc > best_acc:
            best_acc = acc
            file_name = os.path.join(out_dir_base, "model.tar")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": settings.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                file_name,
            )
    return acc, best_acc, summary_writer


def main():
    parser = argparse.ArgumentParser(description="CIFAR LabelDP Training with ALIBI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "mnist", "mnist38", "imdb"],
        help="Dataset to run training on (cifar100 or cifar10)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="wide-resnet",
        choices=["wide-resnet", "resnet", "simple", "simple_conv", "sentiment"],
        help="Resnet-18 architecture (wide-resnet vs resnet)",
    )
    # learning
    parser.add_argument(
        "--bs",
        default=128,
        type=int,
        help="mini-batch size",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="LR momentum")
    parser.add_argument(
        "--weight-decay", default=0.0001, type=float, help="LR weight decay"
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="maximum number of epochs",
    )
    parser.add_argument("--gpu", default=-1, type=int, help="GPU id to use.")
    parser.add_argument(
        "--out-dir-base", type=str, default="/tmp/", help="path to save outputs"
    )
    # Privacy
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "--post-process",
        type=str,
        default="mapwithprior",
        help="Post-processing scheme for noised labels "
        "(MinMax, SoftMax, MinProjection, MAP, MAPWithPrior, RandomizedResponse)",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="Laplace",
        help="Noising mechanism (Laplace or Gaussian)",
    )

    # Attacks
    parser.add_argument(
        "--canary", type=int, default=0, help="Introduce canaries to dataset"
    )

    parser.add_argument("--seed", type=int, default=11337, help="Seed")

    args = parser.parse_args()

    privacy = LabelPrivacy(
        sigma=args.sigma,
        post_process=args.post_process,
        mechanism=args.mechanism,
    )

    learning = Learning(
        lr=args.lr,
        batch_size=args.bs,
        epochs=args.epochs,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        random_aug=False,
    )

    settings = Settings(
        dataset=args.dataset,
        arch=args.arch,
        privacy=privacy,
        learning=learning,
        canary=args.canary,
        gpu=args.gpu,
        out_dir_base=args.out_dir_base,
        seed=args.seed,
    )

    main_worker(settings)


if __name__ == "__main__":
    main()
