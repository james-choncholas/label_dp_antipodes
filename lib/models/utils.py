#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
from torch.optim.lr_scheduler import LambdaLR

from lib.pate.settings import FixmatchModelConfig

def create_model(config: FixmatchModelConfig, num_classes: int):
    if config.model_name == "wideresnet":
        from lib.models.wide_resnet import wideresnet
        return wideresnet(
            depth=config.depth,
            widen_factor=config.width,
            dropout=0,
            num_classes=num_classes,
        )
    elif config.model_name == "simple":
        from lib.models.simple import simple
        return simple(
            num_classes=num_classes
        )
    elif config.model_name == "simple_conv":
        from lib.models.simple import simple_conv
        return simple_conv(
            num_classes=num_classes
        )
    elif config.model_name == "sentiment":
        from lib.models.sentiment import SentimentModel
        return SentimentModel(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            n_layers=config.n_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
        )
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
