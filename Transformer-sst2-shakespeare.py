import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

import re

def parse_sst2_line(line):
    label = int(line[1])
    sentence = ' '.join(re.findall(r'\b[A-Za-z0-9\-\'\.\,]+\b', line))
    return sentence, label


# Load Shakespeare text
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/input.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

# Load SST2 and extract raw sentences
sst2_sentences = []
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            s, _ = parse_sst2_line(line)
            sst2_sentences.append(s)

# Combine all text
combined_text = shakespeare_text + " " + " ".join(sst2_sentences)

# Build vocabulary from combined text
chars = sorted(list(set(combined_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]  # Safe encode
decode = lambda l: ''.join([itos[i] for i in l])







    # here are all the unique characters that occur in this text
# chars = sorted(list(set( shakespeare_text)))
# vocab_size = len(chars)
# # create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode( shakespeare_text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, n_embd):
        super().__init__()
        self.token_embedding_table = pretrained_model.token_embedding_table
        self.position_embedding_table = pretrained_model.position_embedding_table
        self.blocks = pretrained_model.blocks
        self.ln_f = pretrained_model.ln_f
        self.classifier = nn.Linear(n_embd, 1)  # Single logit for binary sentiment

    def forward(self, idx):
        T = idx.shape[1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # [B, n_embd], mean-pooling over sequence
        logit = self.classifier(x).squeeze(-1)  # [B]
        return logit
        
# model = BigramLanguageModel()
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# torch.save(model.state_dict(), 'shakespeare_pretrained.pt')


pretrain_path = 'shakespeare_pretrained.pt'
model = BigramLanguageModel()
m = model.to(device)

if os.path.exists(pretrain_path):
    print("Loading pretrained model weights...")
    model.load_state_dict(torch.load(pretrain_path, map_location=device))
else:
    print("No pretrained model found. Starting pretraining...")
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), pretrain_path)
    print("Pretraining finished and model saved.")



import re

def parse_sst2_line(line):
    label = int(line[1])
    sentence = ' '.join(re.findall(r'\b[A-Za-z0-9\-\'\.\,]+\b', line))
    return sentence, label

sst2_sentences, sst2_labels = [], []
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            s, l = parse_sst2_line(line)
            sst2_sentences.append(s)
            sst2_labels.append(1 if l >= 3 else 0)  # binary label

def encode_and_pad(sentence, block_size):
    ids = encode(sentence)
    if len(ids) < block_size:
        ids += [0] * (block_size - len(ids))  # Pad with zeros
    else:
        ids = ids[:block_size]
    return ids

X = [encode_and_pad(sent, block_size) for sent in sst2_sentences]
Y = sst2_labels
X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Use the already loaded or pretrained model!
classifier_model = SentimentClassifier(model, n_embd).to(device)

optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()  # For binary sentiment

num_epochs = 3  # or as many as you want

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    classifier_model.train()
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = classifier_model(xb)  # [batch]
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        # Accuracy
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, accuracy={correct/total:.4f}")

torch.save(classifier_model.state_dict(), 'finetuned_on_sst2.pt')

    # generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))