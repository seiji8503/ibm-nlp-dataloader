# Import torchtext, a library in PyTorch specialized for NLP (Natural Language Processing)
import torchtext
print(torchtext.__version__)     # Print the version of torchtext (helps in checking compatibility)

# Import basic libraries used in PyTorch and NLP
import pandas as pd # Not used in this snippet, but helpful for data manipulation
from torch.utils.data import Dataset, DataLoader    # Core tools for working with datasets and batching
from torchtext.data.utils import get_tokenizer      # Function to tokenize (split) text into words/tokens
from torchtext.vocab import build_vocab_from_iterator   # To create a vocabulary from tokenized data
from torchtext.datasets import multi30k, Multi30k       # Example NLP dataset (not used here)
from typing import Iterable, List       # Type hints to make code more readable
from torch.nn.utils.rnn import pad_sequence     # For padding sequences (used when sequences are of different lengths)
from torchdata.datapipes.iter import IterableWrapper, Mapper    # Used in more advanced data pipelines (not used here)
import torchtext

# Import PyTorch core libraries
import torch
import torch.nn as nn           # Neural network layers
import torch.optim as optim     # Optimization algorithms (e.g., SGD, Adam)

# Helpful utilities
import numpy as np
import random

# __init__(self, sentences): Initializes the data set with a list of sentences.
# __getitem__(self, idx): Retrieves an item (in this case, a sentence) at a specific index, idx.

# List of example sentences (could be quotes or sample data for NLP tasks)
sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]
# Define your own dataset class by extending PyTorch's Dataset
# Required methods: __init__, __len__, __getitem__
# __getitem__ is called when the DataLoader wants to retrieve one item.


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):

        # Store the list of sentences
        self.sentences = sentences

    def __len__(self):

        # Number of items in the dataset
        return len(self.sentences)

    def __getitem__(self, idx):

        # Return the sentence at the given index
        return self.sentences[idx]

# Create an instance of the dataset using your list of sentences
# Create an instance of your custom dataset
custom_dataset = CustomDataset(sentences)

# Define batch size, Define how many samples (sentences) you want per batch
batch_size = 2

# Create a DataLoader to load data in mini-batches and optionally shuffle the order
# DataLoader is super useful for batching, shuffling, and loading data in parallel.
# batch_size=2 means it will return 2 sentences at a time.

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over each batch and print it
# Each batch is a list of sentences.
# You use batching when training models to process multiple inputs at once (faster and more stable training).

# Iterate through the DataLoader
for batch in dataloader:
    print(batch)

    
