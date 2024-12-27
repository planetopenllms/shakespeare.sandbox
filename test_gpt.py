import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

## local imports
from chartokenizer import CharTokenizer
from chardataset import CharDataset
from gpt import GPTModel, generate_text_simple


"""
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),

"""



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

print( "---\nGPTModel:" )
print( model )

total_params = sum(parameter.numel() for parameter in model.parameters())
print(total_params, "parameters")
#=>  96_864 parameters
## calc size in mega bytes (MBs)
total_bytes = total_params * 4   # assume float32 (4 bytes)
print( f"about  {total_bytes / (1028 *1028) :.2f} MBs, {total_bytes / 1028 :.2f} KBs" )
#=> about  0.366637 MBs, 376.902724 KBs
#=> about  0.37 MBs, 376.90 KBs

model.eval()  # disable dropout


tokenizer=CharTokenizer(chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ3!$&',-.:;? \n")

# construct the training dataset
txt = open('input.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(txt, tokenizer=tokenizer)

# construct the model
vocab_size = train_dataset.get_vocab_size()
block_size = train_dataset.get_block_size()
print( vocab_size, block_size )
#=>  65    vocab size
#=>  128   block size

print( train_dataset )
print( len( train_dataset))

start_context = "O God, O God!"
encoded = tokenizer.encode( start_context )
print( encoded )
# [40, 63, 32, 14, 3, 57, 63, 40, 63, 32, 14, 3, 53]
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print( encoded_tensor )
# tensor([[40, 63, 32, 14,  3, 57, 63, 40, 63, 32, 14,  3, 53]])



### dump summary via torchsummary -- sorry not working for now
## from torchsummary import summary
# Print the summary
# print( "input_size", encoded_tensor.shape, encoded_tensor.ndim )
# summary(model, input_size=encoded_tensor.shape)
# summary(model, input_size=(13,))


print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_NANO["context_length"]
    )

decoded_text = tokenizer.decode( out.squeeze(0).tolist() )

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out)
print("Output length:", len(out[0]))
print("Output text:", decoded_text)


print("bye")
