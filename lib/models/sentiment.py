#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        # GlobalAveragePooling1D
        # Sum along the sentence length dimension
        summed = embedded.sum(dim=1)
        # summed = [batch size, emb dim]
        
        # Divide by the actual lengths of each sentence
        # text_lengths needs to be reshaped to [batch size, 1] to broadcast correctly
        pooled = summed / text_lengths.unsqueeze(1).float()
        # pooled = [batch size, emb dim]

        dropped_out = self.dropout(pooled)
        
        return self.fc(dropped_out)
