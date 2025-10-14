#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from lib.dataset.imdb_custom import get_imdb_custom
import random
from torchtext import data
import torch
from torch.utils.data import Subset
 
class ImdbSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.fields = dataset.fields
 
def get_imdb(root: str, student_seed: int):
    # For reproducibility
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
 
    TEXT = data.Field(tokenize='spacy',
                       tokenizer_language='en_core_web_sm',
                       lower=True,
                       include_lengths=True,
                       batch_first=True)
    
    LABEL = data.LabelField(dtype=torch.long)

    train_data_raw, test_data_raw = get_imdb_custom(root)

    train_examples = [data.Example.fromlist([text, label], [('text', TEXT), ('label', LABEL)]) for text, label in train_data_raw]
    # test_examples = [data.Example.fromlist([text, label], [('text', TEXT), ('label', data.LabelField(dtype=torch.float))]) for text, label in test_data_raw]
    test_examples = [data.Example.fromlist([text, label], [('text', TEXT), ('label', LABEL)]) for text, label in test_data_raw]


    train_data = data.Dataset(train_examples, [('text', TEXT), ('label', LABEL)])
    # Recreate LABEL field for test data to maintain original behavior
    # TEST_LABEL = data.LabelField(dtype=torch.float)
    test_data = data.Dataset(test_examples, [('text', TEXT), ('label', LABEL)])

    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    
    student_data, _ = train_data.split(random_state=random.seed(student_seed), split_ratio=0.1)


    # MAX_VOCAB_SIZE = 25_000
    MAX_VOCAB_SIZE = 10000-5

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    # TEST_LABEL.build_vocab(test_data)

    return {
        "labeled": train_data,
        "unlabeled": valid_data,
        "test": test_data,
        "student": student_data,
        "text": TEXT,
        "label": LABEL,  # This now refers to the one-hot field for train/valid
        # "test_label": TEST_LABEL # Separate field for test
    }

def get_imdb_dataloaders(root: str, batch_size: int, student_seed: int, device: torch.device):
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    datasets = get_imdb(root, student_seed)
    train_data = datasets["labeled"]
    valid_data = datasets["unlabeled"]
    test_data = datasets["test"]
    TEXT = datasets["text"]
    LABEL = datasets["label"]

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device
    )
    
    return {
        "labeled": train_iterator,
        "unlabeled": valid_iterator,
        "test": test_iterator,
        "student": None, # Student iterator can be created if needed
        "vocab": TEXT.vocab,
        "pad_idx": TEXT.vocab.stoi[TEXT.pad_token],
        "LABEL": LABEL,
        "TEXT": TEXT
    }