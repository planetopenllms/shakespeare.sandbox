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




def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss



def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()



def train_model_simple(model, train_loader, val_loader, optimizer,
                       num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} "
                      f"Token seen {tokens_seen}")
                generate_and_print_sample( model, tokenizer, start_context )

        # Print a sample text after each epoch
        generate_and_print_sample( model, tokenizer, start_context )

    return train_losses, val_losses, track_tokens_seen



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()




### calc loss on first five
print( "\n==> test calc loss" )
for i, (x,y) in enumerate(train_loader):
   loss = calc_loss_batch( x, y, model )
   print( "loss", i+1, loss.item() )
   if i > 5:
      break


### setup training

lr           = 0.0004
weight_decay = 0.1

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr          = lr,
    weight_decay= weight_decay
)

num_epochs = 3
start_context = "O God, O God!"

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer,
    num_epochs=num_epochs,
    eval_freq=5, eval_iter=1,
    start_context=start_context,
    tokenizer=tokenizer
)

print( "train_losses", train_losses )
print( "val_losses", val_losses )
print( "tokens_seen", token_seen )


print("bye")


"""
Ep 1 (Step 000000): Train loss 4.260, Val loss 4.317
Ep 1 (Step 000005): Train loss 4.106, Val loss 4.208
Ep 1 (Step 000010): Train loss 3.960, Val loss 4.100
Ep 1 (Step 000015): Train loss 3.809, Val loss 3.996
Ep 1 (Step 000020): Train loss 3.686, Val loss 3.903
Ep 1 (Step 000025): Train loss 3.584, Val loss 3.824
Ep 1 (Step 000030): Train loss 3.502, Val loss 3.759
Ep 1 (Step 000035): Train loss 3.433, Val loss 3.702
Ep 1 (Step 000040): Train loss 3.394, Val loss 3.653
Ep 1 (Step 000045): Train loss 3.352, Val loss 3.612
Ep 1 (Step 000050): Train loss 3.321, Val loss 3.579
Ep 1 (Step 000055): Train loss 3.331, Val loss 3.553
"""
