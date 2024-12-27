"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

## local imports
from chartokenizer import CharTokenizer




## todo - change block_size to max_length -  why? why not?

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, text, tokenizer, block_size=128):
        self.block_size = block_size
        self.tokenizer  = tokenizer
        self.text        = text

        print('data has %d characters, %d unique.' % (len(text),
                                                      self.tokenizer.get_vocab_size()))

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.text) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.text[idx:idx + self.block_size + 1]
        # encode every character to an integer
        encoded = self.tokenizer.encode( chunk )
        # return as tensors
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y



def dump_item( item, name=None ):
    if name:
        print( "==>", name )
    x, y = item
    print( "x", type(x), x.shape, x.ndim )
    print( "y", type(y), y.shape, y.ndim )
    print( item )




if __name__ == '__main__':

    shakespeare_chars =  'abcdefghijklmnopqrstuvwxyz' + \
                         'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
                         '3' + \
                          '!$&\',-.:;?' + \
                         ' \n'

    print(len(shakespeare_chars), shakespeare_chars)
    # "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ3!$&',-.:;?"

    tokenizer = CharTokenizer( chars=shakespeare_chars )


    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(text,tokenizer=tokenizer)

    # construct the model
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_block_size()
    print( vocab_size, block_size )
    #=>  65    vocab size
    #=>  128   block size


    print( train_dataset )
    print( len( train_dataset))

    dump_item( train_dataset[0], name="0"  )
    dump_item( train_dataset[1], name="1"  )

    ## note negative index will not work !!!!
    ##   returns []
    dump_item( train_dataset[-1], name="-1" )
    dump_item( train_dataset[-2], name="-2" )

    train_dataset_len = len( train_dataset)

    dump_item( train_dataset[train_dataset_len-1], name="train_dataset_len-1")
    dump_item( train_dataset[train_dataset_len-2], name="train_dataset_len-2")

    ## check index beyond length
    ## will result in smaller tensor (128 -> 127/126/125/etc.)
    dump_item( train_dataset[train_dataset_len], name="train_dataset_len")
    dump_item( train_dataset[train_dataset_len+1], name="train_dataset_len+1")
    dump_item( train_dataset[train_dataset_len+2], name="train_dataset_len+2")

    print("bye")

