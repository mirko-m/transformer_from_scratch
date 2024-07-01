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
    
class Transformer(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.pos_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.weights_k = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weights_q = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weights_v = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.weights_o = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        # x: (B, T)
        _, seq_length = x.shape
        tril = torch.tril(torch.ones((seq_length, seq_length)))
        x = self.embedding(x) # (B, T, H)
        x += self.pos_embedding(torch.arange(0, seq_length)) # (B, T, H)

        k = self.weights_k(x) # (B, T, H)
        q = self.weights_q(x) # (B, T, H)
        v = self.weights_v(x) # (B, T, H)

        z = q @ k.transpose(1,2) # (B, T, T) the attention matrix
        z /= self.hidden_size**0.5
        z = z.masked_fill(tril, float("-inf"))
        z = F.softmax(z, dim=2)
        z = tril * z # (B, T, T) mask out non-causal (q, k) pairs
        z = z @ v # (B, T, H) the values
        z = z + x # (B, T, H) residual connection
        z = self.weights_o(z) # (B, T, V) where V is vocab_size, logits

        return z