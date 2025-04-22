# q:  can you generate a python script with pytorch that builds / trains a char-level language model using a transformer and trains on the complete work of shakespeare?

Absolutely! Below is a Python script that uses PyTorch to build and train a character-level language model using a Transformer architecture on the complete works of Shakespeare.

This example:
- Uses a simple Transformer model.
- Trains on character-level data.
- Loads Shakespeare text from a file (`shakespeare.txt`).

First, make sure you have the necessary packages:
```bash
pip install torch tqdm
```

Now, here's the Python script:  
(Note: Save your Shakespeare text as `shakespeare.txt` in the same directory.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# Load and encode data
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Hyperparameters
block_size = 128
batch_size = 64
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
class CharDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size]
        y = self.data[idx+1:idx+block_size+1]
        return x, y

train_loader = DataLoader(CharDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CharDataset(val_data), batch_size=batch_size)

# Transformer model
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = TransformerLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
def evaluate():
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

for step in range(max_iters):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # only one batch per step

    if step % eval_interval == 0 or step == max_iters - 1:
        val_loss = evaluate()
        print(f"Step {step}: val loss = {val_loss:.4f}")

# Generate sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

---

Let me know if you want this turned into a Colab notebook, or if you want training speed optimizations or inference tweaks!


---

## q: can you explain the numbers for your hyperparameter choices: n_embd = 256, n_head = 8, n_layer = 6

You're welcome! And great question — choosing those hyperparameters isn't totally arbitrary, but it's also often a mix of intuition, past research, and practical trade-offs. Let's walk through each one:

---

### 🔢 `n_embd = 256` — **Embedding Size**

This is the dimensionality of the token and position embeddings.

- **Why 256?**
  - It strikes a **balance between capacity and speed/memory use**.
  - Bigger embeddings (e.g., 512, 1024+) allow the model to represent richer relationships between characters, but they also **consume more memory and make training slower**.
  - Since this is **character-level modeling**, we don't need massive embeddings like in token-level models trained on large vocabularies (like GPT-3). Characters are simpler building blocks.
  - 256 is commonly used in smaller-scale models and gives **decent performance** on datasets like Shakespeare.

---

### 👥 `n_head = 8` — **Number of Attention Heads**

Multi-head attention allows the model to focus on different subspaces of the embedding.

- **Why 8?**
  - The embedding size `n_embd` must be divisible by `n_head`. Here, 256 / 8 = 32 dimensions per head.
  - 8 is often a **sweet spot**:
    - Enough heads to allow **diverse attention patterns**.
    - Not so many that each head becomes too narrow or inefficient.
  - For larger models (e.g., 512 or 768 embedding size), people often use 8–12 heads. But 8 works very well for 256-dimensional embeddings.

---

### 🧱 `n_layer = 6` — **Number of Transformer Blocks**

Each layer is a Transformer block (attention + feedforward + residuals).

- **Why 6?**
  - More layers = **more abstraction and depth**, but also **more computation and risk of overfitting**.
  - 6 layers is:
    - **Deep enough** to model long-term dependencies.
    - Still **manageable on a GPU (or even CPU with patience)**.
  - GPT-2 Small has 12 layers, but it works with token-level inputs and a much larger vocabulary. For **char-level on a smaller dataset**, 6 layers is a solid starting point.

---

### TL;DR:

| Hyperparam | Value | Why it works |
|------------|-------|--------------|
| `n_embd`   | 256   | Enough to encode rich char-level info; light on memory. |
| `n_head`   | 8     | Good diversity in attention, efficient split of 256 dim. |
| `n_layer`  | 6     | Deep enough to model language patterns; not overkill.   |

---

If you're curious, you can try these tweaks to explore trade-offs:
- Try `n_embd=128` for faster training (but maybe less fluent output).
- Try `n_layer=2` to see how shallow models perform.
- Try `n_embd=512, n_layer=8, n_head=8` if you have a GPU and want to push quality.

Want help visualizing how these affect model performance or training curves?



