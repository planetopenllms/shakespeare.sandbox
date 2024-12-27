import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

## local imports
from chartokenizer import CharTokenizer
from chardataset import CharDataset




def create_dataloaders(  txt,
                         tokenizer,
                         batch_size=4,
                         max_length=128 ):

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(txt))

    train_dataloader = DataLoader(
        CharDataset(txt[:split_idx], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, num_workers=0)

    val_dataloader = DataLoader(
        CharDataset(txt[split_idx:], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, num_workers=0)


    return train_dataloader, val_dataloader





### test drive
torch.manual_seed(123)

tokenizer=CharTokenizer(chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ3!$&',-.:;? \n")

# construct the training dataset
txt = open('input.txt', 'r').read() # don't worry we won't run out of file handles


batch_size     = 100
context_length = 128

train_loader, val_loader  = create_dataloaders(
                              txt,
                              tokenizer=tokenizer,
                              batch_size=batch_size,
                              max_length=context_length )


print( "train_loader - num_batches", len( train_loader) )
train_iter = iter(train_loader)
x, y = next(train_iter)
print( "x", x.ndim, x )
print( "y", y.ndim, y )

print( "val_loader - num_batches", len( val_loader) )
val_iter = iter(val_loader)
x, y = next(val_iter)
print( "x", x.ndim, x )
print( "y", y.ndim, y )

# train_loader - num_batches 10037
# val_loader - num_batches 1115


print("Train loader:")
for i, (x, y) in enumerate(train_loader):
    print( i, x.shape, y.shape )
#  7839 torch.Size([128, 128]) torch.Size([128, 128])
#   7840 torch.Size([128, 128]) torch.Size([128, 128])
    print( "x", x.ndim, x )
    print( "y", y.ndim, y )
    break  # on first

"""
x 2 tensor([[ 4, 12,  8,  ...,  8, 18, 63],
        [63, 14,  5,  ..., 24, 63,  2],
        [48,  8,  5,  ..., 14, 11, 18],
        ...,
        [15,  4,  0,  ..., 19,  7,  4],
        [14, 62, 64,  ..., 39, 28, 30],
        [29, 63, 34,  ..., 19, 24, 63]])
y 2 tensor([[12,  8,  0,  ..., 18, 63,  5],
        [14,  5, 63,  ..., 63,  2,  7],
        [ 8,  5,  4,  ..., 11, 18,  2],
        ...,
        [ 4,  0, 10,  ...,  7,  4, 63],
        [62, 64, 34,  ..., 28, 30, 39],
        [63, 34, 34,  ..., 24, 63, 19]])
"""

print("\nValidation loader:")
for i, (x, y) in enumerate(val_loader):
    print( i, x.shape, y.shape )
    print( "x", x.ndim, x )
    print( "y", y.ndim, y )
    break  # on first

#  869 torch.Size([128, 128]) torch.Size([128, 128])
#  870 torch.Size([52, 128]) torch.Size([52, 128])

print("bye")
