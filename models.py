import torch
from torch import nn
from torch.nn import functional as F

class BigramModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, x):
        logits = self.embedding(x)
        return logits
    
    def generate(self, idx, max_length):
        # idx is a tensor of indexes with shape (B, T)
        assert max_length > 0, "max_length must be at least 1"
        # The bigram does not use the context, just the last characters of the sequence.
        # Because of that we can use idx[:, -1]
        sampled = idx[:,-1] # (B)
        out = []
        while max_length > 0:
            logits = self.forward(sampled) # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            sampled = torch.multinomial(probs, 1, replacement=True) # (B, 1)
            sampled = sampled.flatten() # (B)
            out.append(sampled)
            max_length -= 1
            
        return torch.cat((idx, torch.stack(out, dim=1)), dim=1)
    
class AttentionHead(nn.Module):

    def __init__(self, input_size: int, head_size: int):
        super().__init__()
        self.input_size = input_size
        self.head_size = head_size

        self.weights_k = torch.nn.Linear(self.input_size, self.head_size, bias=False)
        self.weights_q = torch.nn.Linear(self.input_size, self.head_size, bias=False)
        self.weights_v = torch.nn.Linear(self.input_size, self.head_size, bias=False)


    def forward(self, x):
        # x: (B, T, H)
        seq_length = x.shape[1]
        tril = torch.tril(torch.ones((seq_length, seq_length)))

        k = self.weights_k(x) # (B, T, H/n) where H/n is head_size
        q = self.weights_q(x) # (B, T, H/n)
        v = self.weights_v(x) # (B, T, H/n)

        z = q @ k.transpose(1,2)/self.head_size**0.5 # (B, T, T) the attention matrix
        z = z.masked_fill(tril==0, float("-inf"))
        z = F.softmax(z, dim=2)
        z = z @ v # (B, T, H/n) the values

        return z

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int):
        assert hidden_size % num_heads == 0, "num_heads must divide hidden_size"
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = self.hidden_size // num_heads

        self.heads = [AttentionHead(self.hidden_size, self.head_size) for _ in range(self.num_heads)]
        self.weights_o = nn.Linear(self.hidden_size, self.hidden_size)

        ff_size = self.hidden_size*4
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, self.hidden_size)
        )

    def forward(self, x):
        # x: (B, T, H)
        zs = []
        for head in self.heads:
            zs.append(head(x)) # (B, T, H/n)
        z = torch.cat(zs, dim=2) # (B, T, H)
        z = self.weights_o(z) # (B, T, H) 
        z += x # (B, T, H) 

        z = self.ff(z) + z # (B, T, H) 

        return z

class Transformer(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int):
        assert hidden_size % num_heads == 0, "num_heads must divide hidden_size"
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = self.hidden_size // self.num_heads

        self.pos_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.block = TransformerBlock(self.hidden_size, self.num_heads)
        self.weights_proj = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        # x: (B, T)
        _, seq_length = x.shape
        x = self.embedding(x) # (B, T, H)
        x += self.pos_embedding(torch.arange(0, seq_length)) # (B, T, H)
        z = self.block(x) # (B, T, H)
        z = self.weights_proj(z) # (B, T, V) where V is vocab_size, logits
        return z