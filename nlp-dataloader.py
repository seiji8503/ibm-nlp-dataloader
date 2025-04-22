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

"""""
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
"""
    
############ Creating Tensors #############

# - __init__: The constructor takes a list of sentences, a tokenizer function, and a vocabulary (vocab) as input.
# - __len__: This method returns the total number of samples in the data set.
# - __getitem__: This method is responsible for processing a single sample. It tokenizes the sentence using the provided tokenizer and then converts the tokens into tensor indices using the vocabulary.
    

"""""
# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch
"""

def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    
    # Return the padded batch
    return padded_batch

# Create a custom collate function
def collate_fn_bfFALSE(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, padding_value=0)
    return padded_batch

"""
# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)
"""

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set
# custom_dataset = CustomDataset(sentences, tokenizer, vocab)

"""
print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")
"""


# Create an instance of your custom data set
# custom_dataset = CustomDataset(sentences, tokenizer, vocab)

# Define batch size
batch_size = 2

# Create a data loader
#dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

"""
# Iterate through the data loader
for batch in dataloader:
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)
    

# Create a data loader with the custom collate function with batch_first=True,
dataloader_bfFALSE = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn_bfFALSE)

# Iterate through the data loader
for seq in dataloader_bfFALSE:
    for row in seq:
        #print(row)
        words = [vocab.get_itos()[idx] for idx in row]
        print(words)
"""            

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset=CustomDataset(sentences)

# Create a data loader for the custom dataset
dataloader = DataLoader(
    dataset=custom_dataset,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=True,              # Shuffle the data at the beginning of each epoch
    collate_fn=collate_fn      # Custom collate function for processing batches
)

"""
for batch in dataloader:
    print(batch)
    print("shape of sample",len(batch))
"""    


corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]
for batch in dataloader:
    print(batch)

def collate_fn_french(batch):
    
    # Pad sequences within the batch to have equal lengths using pad_sequence

    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))
    
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    
    # Return the padded batch
    return padded_batch

# Tokenizer
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, corpus))

# Sort sentences based on their length
sorted_data = sorted(corpus, key=lambda x: len(tokenizer(x)))
#print(sorted_data)
dataloader = DataLoader(sorted_data, batch_size=4, shuffle=False, collate_fn=collate_fn_french)

for batch in dataloader:
    print(batch)