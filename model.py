# if im aware this could generate solid english if trained with enough steps.
# this was an experiment to see how transformer models work. this was made following a tutorial from youtube and docs from pytorch c:

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from datasets import load_dataset
import json
import os

batch_size = 48
block_size = 128
iterations = 10000
learning_rate = 3e-4
n_embed = 128
n_head = 4
n_layer = 4
dropout = 0.1
dataset_name = "vesteinn/babylm"
TOKENIZER_FILE = "tokenizer_char.json"
DATA_FILE = "data.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

try:
    # if ur training on google colab's tpu
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except (ImportError, RuntimeError):
    pass

print("Using device:", device)

print("Using device:", device)

# --- Tokenizer (Character-Level) ---
class Tokenizer:
    def __init__(self, chars=None):
        if chars:
            self.chars = chars
            self.vocab_size = len(self.chars)
            self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx.get(char, -1) for char in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(token, '') for token in tokens])
        
    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f)
    
    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(chars=data['chars'])

def get_batch(data, batch_size, block_size, device):
    # loading data into the deviec
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

def main():
    if os.path.exists(TOKENIZER_FILE) and os.path.exists(DATA_FILE):
        print("Already tokenized, loading from file")
        tokenizer = Tokenizer.load(TOKENIZER_FILE)
        data = torch.load(DATA_FILE)
    else:
        print("Tokenizing data from scratch...")
        dataset = load_dataset(dataset_name, split="train")
        all_text = "\n".join([sample['text'] for sample in dataset if sample['text']])
        
        chars = sorted(list(set(all_text)))
        tokenizer = Tokenizer(chars)
        
        data = torch.tensor(tokenizer.encode(all_text), dtype=torch.long)

        print("Saving tokenized data...")
        tokenizer.save(TOKENIZER_FILE)
        torch.save(data, DATA_FILE)
    
    print(f"Dataset loaded, voacb size is {tokenizer.vocab_size}")

    # my idiotic self used rmsprop for optimizing and that lead to HORRIBLE results
    model = TransformerModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # checking if its running on google colabs tpu
    is_xla_device = 'xla' in str(device)
    
    print(f"Starting training for {iterations} iterations...")
    for step in trange(iterations):
        xb, yb = get_batch(data, batch_size, block_size, device)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # figured out that xm has an optimizer seperated from the other one... (this cost me 1 hour)
        if is_xla_device:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
        
        if (step + 1) % 200 == 0:
            print(f"Step {step+1}/{iterations}, Loss: {loss.item():.6f}")

    # sampling text
    print("\nTraining finished. Generating text...")
    start_char = "\n" if "\n" in tokenizer.char_to_idx else list(tokenizer.char_to_idx.keys())[0]
    start_token = tokenizer.encode(start_char)[0]
    context = torch.tensor([[tokenizer.char_to_idx['\n']]], dtype=torch.long, device=device)

    generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
    print("Generated text:")
    print(tokenizer.decode(generated_tokens))

if __name__ == "__main__":
    main()
