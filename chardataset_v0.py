"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader



class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, data, block_size=128):
        self.block_size = block_size

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y



def dump_item( item, name=None ):
    if name:
        print( "==>", name )
    x, y = item
    print( "x", type(x), x.shape, x.ndim )
    print( "y", type(y), y.shape, y.ndim )
    print( item )




if __name__ == '__main__':


    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(text)

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



"""
    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

"""