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