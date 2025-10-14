#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import tarfile
import urllib.request
from pathlib import Path

def get_imdb_custom(root):
    """
    Downloads and extracts the IMDB dataset if it's not already present.
    Then, loads the data from the text files and returns it.
    """
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_path = Path(root) / "aclImdb"

    if not dataset_path.exists():
        print("Downloading and extracting IMDB dataset...")
        os.makedirs(root, exist_ok=True)
        tar_path = Path(root) / "aclImdb_v1.tar.gz"
        
        # Download
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=root)
        
        # Clean up
        os.remove(tar_path)
        print("Dataset downloaded and extracted.")

    train_data = []
    test_data = []

    # Load training data
    for label in ["pos", "neg"]:
        train_dir = dataset_path / "train" / label
        for file in train_dir.iterdir():
            with open(file, "r", encoding="utf-8") as f:
                train_data.append((f.read(), label))

    # Load testing data
    for label in ["pos", "neg"]:
        test_dir = dataset_path / "test" / label
        for file in test_dir.iterdir():
            with open(file, "r", encoding="utf-8") as f:
                test_data.append((f.read(), label))

    return train_data, test_data