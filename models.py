import torch
from torch import nn
from torch.nn import functional as F

class BigramModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, x: torch.Tensor):
        logits = self.embedding(x)
        return logits
    
    def generate(self, idx: torch.Tensor, max_length: int) -> torch.Tensor:
        # idx is a tensor of indexes with shape (B, T)
        assert max_length > 0, "max_length must be at least 1"
        # The bigram does not use the context, just the last characters of the sequence.
        # Because of that we can use idx[:, -1]
        sampled = idx[:,-1] # (B)
        out = []
        for _ in range(max_length):
            logits = self.forward(sampled) # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            sampled = torch.multinomial(probs, 1, replacement=True) # (B, 1)
            sampled = sampled.flatten() # (B)
            out.append(sampled)
            
        return torch.cat((idx, torch.stack(out, dim=1)), dim=1)
    
class AttentionHead(nn.Module):

    def __init__(self, input_size: int, head_size: int, context_length: int):
        super().__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.context_length = context_length

        self.weights_k = nn.Linear(input_size, head_size, bias=False)
        self.weights_q = nn.Linear(input_size, head_size, bias=False)
        self.weights_v = nn.Linear(input_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((context_length, context_length))))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        seq_length = x.shape[1]

        k = self.weights_k(x) # (B, T, H/n) where H/n is head_size
        q = self.weights_q(x) # (B, T, H/n)
        v = self.weights_v(x) # (B, T, H/n)

        attn = q @ k.transpose(1,2)/self.head_size**0.5 # (B, T, T) the attention matrix
        attn = attn.masked_fill(self.tril[:seq_length,:seq_length]==0, float("-inf"))
        attn = F.softmax(attn, dim=2)
        z = attn @ v # (B, T, H/n) the values

        return z
    
class MultiAttentionHead(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, context_length: int):
        assert hidden_size % num_heads == 0, "num_heads must divide hidden_size"
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = self.hidden_size // num_heads
        self.context_length = context_length
        
        self.heads = nn.ModuleList([
            AttentionHead(hidden_size, self.head_size, context_length) for _ in range(num_heads)
        ])
        self.weights_o = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        zs = []
        x_ = self.ln(x)
        for head in self.heads:
            zs.append(head(x_)) # (B, T, H/n)
        z = torch.cat(zs, dim=2) # (B, T, H)
        z = self.weights_o(z) # (B, T, H)
        return z

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, context_length: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_length = context_length

        self.multi_head = MultiAttentionHead(hidden_size, num_heads, context_length)
        ff_size = self.hidden_size*4
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, self.hidden_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        z = x + self.dropout1(self.multi_head(x)) # (B, T, H)
        z = z + self.dropout2(self.ff(self.ln(z))) # (B, T, H)
        return z

class Transformer(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_blocks,
                 context_length: int, dropout: float):
        assert hidden_size % num_heads == 0, "num_heads must divide hidden_size"
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = self.hidden_size // self.num_heads
        self.num_blocks = num_blocks
        self.context_length = context_length

        self.pos_embedding = nn.Embedding(context_length, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_size, num_heads, context_length, dropout) for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(hidden_size)
        self.weights_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        _, seq_length = x.shape
        x = self.embedding(x) # (B, T, H)
        x += self.pos_embedding(torch.arange(0, seq_length, device=x.device)) # (B, T, H)
        x = self.dropout(x)
        x = self.blocks(x) # (B, T, H)
        x = self.ln(x) # (B, T, H)
        x = self.weights_proj(x) # (B, T, V) where V is vocab_size, logits
        return x

    def generate(self, idx: torch.Tensor, max_length: int) -> torch.Tensor:
        # idx is a tensor of indexes with shape (B, T)
        assert max_length > 0, "max_length must be at least 1"
        for _ in range(max_length):
            logits = self.forward(idx) # (B, T, V)
            sample = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1) # (B, 1)
            idx = torch.cat((idx, sample), dim=1)
        return idx
