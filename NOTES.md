

## search

search
- gpt shakespeare training
- gpt shakespeare training from scratch

- add character-level - why? why not?


links


- https://github.com/shreydan/shakespeareGPT.git




- https://gist.github.com/s-casci/0bad1a671d37d52ada3fb514046103ba

```
class Tokenizer:
    def __init__(self, text):
        tokens = list(set(text))
        self.chars_to_tokens = {t: i for (i, t) in enumerate(tokens)}
        self.tokens_to_chars = {i: t for (i, t) in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def encode(self, text: str) -> List[str]:
        return [self.chars_to_tokens[char] for char in text]

    def decode(self, tokens: List[str]) -> str:
        return "".join([self.tokens_to_chars[token] for token in tokens])

...


def main(
    block_size: int = 32,
    num_embeddings: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    learning_rate: float = 1e-3,
    num_iterations: int = 5000,
    batch_size: int = 16,
    generate_every: int = 500,
    max_new_tokens: int = 500,
):
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    model = CausalTransformer(
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        num_embeddings=num_embeddings,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    print(sum(parameter.numel() for parameter in model.parameters()) / 1e6, "M parameters")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def get_batch(batch_size: int) -> Tuple[torch.tensor, torch.tensor]:
        indices = torch.randint(len(data) - block_size, (batch_size,))
        batch_xs = torch.stack([data[index : index + block_size] for index in indices]).to(model.device)
        batch_ys = torch.stack([data[index + 1 : index + block_size + 1] for index in indices]).to(model.device)
        return batch_xs, batch_ys

    @torch.no_grad
    def generate(max_new_tokens: int) -> str:
        tokens = tokenizer.encode("\n")
        model.eval()
        for _ in range(max_new_tokens):
            preds = model(
                torch.tensor(
                    [tokens[-block_size:]],
                    dtype=torch.long,
                    device=model.device,
                )
            )
            logits = preds[0][-1]
            probs = F.softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
        model.train()
        return tokenizer.decode(tokens)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=num_iterations)

    model.train()
    for it in range(1, num_iterations + 1):
        batch_xs, batch_ys = get_batch(batch_size)
        preds = model(batch_xs)
        loss = F.cross_entropy(
            preds.view(-1, tokenizer.vocab_size),
            batch_ys.view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"it: {it:4d}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        if it % generate_every == 0:
            print(f"\n\n{generate(max_new_tokens)}\n\n")

    model.eval()
    print("\nFinal generation:")
    print(generate(max_new_tokens * 10))

```




Training nanoGPT entirely on content from my blog
- https://til.simonwillison.net/llms/training-nanogpt-on-my-blog
- https://til.simonwillison.net/llms/nanogpt-shakespeare-m2

```
I ran the train.py script in the repository root like this:

 python train.py \
  --dataset=simonwillisonblog \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=mps
The --dataset option points to my new simonwillisoblog folder. --device=mps is needed for the M2 MacBook Pro - using --device=cpu runs about 3 times slower.

I ran this for iter 20,143 iterations before hitting Ctrl+C to stop it.

The script writes out a model checkpoint every 2,000 iterations. You can modify the eval_interval= variable in the script to change that - I suggest switching it to something lower like 200, since then you can try sampling the model more frequently while it trains.

Getting to 20,000 iterations took around 45 minutes.
```

```
Next change back up to the nanoGPT directory and run the command to train the model:

time python train.py \
  --dataset=shakespeare \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=cpu
```




- https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html

```
After setting these, we load the Tiny Shakespeare dataset and print the length
of the training set, which is around one million characters,
and that of the validation set (around 50k characters). Finally, we print a small snippet of the train set.

# @markdown Random seed:
SEED = 42  # @param{type:"integer"}
# @markdown Learning rate passed to the optimizer:
LEARNING_RATE = 5e-3 # @param{type:"number"}
# @markdown Batch size:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Number of training iterations:
N_ITERATIONS = 50_000  # @param{type:"integer"}
# @markdown Number of training iterations between two consecutive evaluations:
N_FREQ_EVAL = 2_000 # @param{type:"integer"}
# @markdown Rate for dropout in the transformer model
DROPOUT_RATE = 0.2  # @param{type:"number"}
# @markdown Context window for the transformer model
BLOCK_SIZE = 64  # @param{type:"integer"}
# @markdown Number of layer for the transformer model
NUM_LAYERS = 6  # @param{type:"integer"}
# @markdown Size of the embedding for the transformer model
EMBED_SIZE = 256  # @param{type:"integer"}
# @markdown Number of heads for the transformer model
NUM_HEADS = 8  # @param{type:"integer"}
# @markdown Size of the heads for the transformer model
HEAD_SIZE = 32  # @param{type:"integer"}

vocab = sorted(list(set(text_train)))
print("Vocabulary:, ", "".join(vocab))
print("Length of vocabulary: ", len(vocab))
Vocabulary:,
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
Length of vocabulary:  65

class NanoLM(nn.Module):
  """NanoLM model."""
  vocab_size: int
  num_layers: int = 6
  num_heads: int = 8
  head_size: int = 32
  dropout_rate: float = 0.2
  embed_size: int = 256
  block_size: int = 64


```

- https://lambdalearner.com/the-worst-shakespeare-ever-written-how-i-trained-a-gpt-model/
  - https://github.com/chibeze01/transformer-from-scratch/blob/master/prepare.py
  - https://github.com/chibeze01/transformer-from-scratch/blob/master/train.py
  - https://github.com/chibeze01/transformer-from-scratch/blob/master/TransformerLm.py

```
device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_type)
data_dir = os.path.join(os.path.dirname(__file__))
batch_size = 64
block_size = 256
max_iters = 5000
eval_iters = 500
n_embd = 384
n_heads = 6
dropout = 0.2
n_layers = 6
learning_rate = 3e-4
torch.manual_seed(1337)

```

more

Shakespearean GPT from scratch: create a generative pre-trained transformer
- https://community.wolfram.com/groups/-/m/t/2847286

gpt2-shakespeare -
This model is a fine-tuned version of gpt2 on datasets containing Shakespeare Books.
- https://huggingface.co/sadia72/gpt2-shakespeare





## chargpt (by Andrej Karpathy)

chargpt trains a character-level language model.

- a user specified `input.txt` file that we train an LM on (e.g. get tiny-shakespear (1.1MB of data) [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))
