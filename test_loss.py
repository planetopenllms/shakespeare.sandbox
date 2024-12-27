import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

## local imports
from chartokenizer import CharTokenizer
from chardataset import CharDataset
from gpt import GPTModel, generate_text_simple



GPT_CONFIG_NANO = {
        "vocab_size": 65,       # Vocabulary size
        "context_length": 128,  # Context length / max length / block size
        "emb_dim": 48,          # Embedding dimension
        "n_heads": 3,           # Number of attention heads
        "n_layers": 3,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
}


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_NANO)


model.eval()  # disable dropout




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
tokenizer=CharTokenizer(chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ3!$&',-.:;? \n")

# construct the training dataset
txt = open('input.txt', 'r').read() # don't worry we won't run out of file handles

batch_size = 100

train_loader, val_loader  = create_dataloaders(
                              txt,
                              tokenizer=tokenizer,
                              batch_size=batch_size,
                              max_length=GPT_CONFIG_NANO['context_length'] )




def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
                            logits.flatten(0, 1),
                            target_batch.flatten())
    return loss


### calc loss on first five
print( "\n==> calc loss" )
train_iter = iter(train_loader)
input_batch, target_batch = next(train_iter)   # x,y
print( "input_batch", input_batch.shape, input_batch )
print( "target_batch", target_batch.shape, target_batch )

loss = calc_loss_batch( input_batch, target_batch, model )
print( "loss", loss.item() )

for i, (x,y) in enumerate(train_loader):
   loss = calc_loss_batch( x, y, model )
   print( "loss", i+1, loss.item() )
   if i > 5:
      break



print("bye")
