import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F 
from torch.optim import Optimizer
from collections.abc import Iterable

class TextData(Dataset):

    def __init__(self, filepath: str, seq_length: int, train: bool, train_frac: float = 0.9):

        self.filepath = filepath
        self.seq_length = seq_length
        self.train = train
        self.train_frac = train_frac

        with open(filepath, "r", encoding="utf-8") as fp:
            self.text = fp.read()
        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        i_split = int(train_frac*data.shape[0])
        if self.train:
            self.data = data[:i_split]
        else:
            self.data = data[i_split:]
        self.data_size = self.data.shape[0]

    def encode(self, text_sample: str) -> list:
        return [self.char_to_idx[c] for c in text_sample]
    
    def decode(self, idx: Iterable) -> str:
        return "".join(self.idx_to_char[int(i)] for i in idx)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx: idx+self.seq_length]
        y = self.data[idx+1: idx+self.seq_length+1]
        return x, y
    
def get_batch(batch_size: int, dataset: TextData) -> torch.Tensor:
    # FIXME: add to device here?
    idx = torch.randint(0, dataset.data_size - dataset.seq_length, (batch_size,))
    xs = []
    ys = []
    for i in idx:
        x_, y_ = dataset[i]
        xs.append(x_)
        ys.append(y_)
    x = torch.stack(xs)
    y = torch.stack(ys)
    return x, y

def evaluate(model: nn.Module, val_dataset: TextData, batch_size: int, num_samples: int) -> float:
    seq_length = val_dataset.seq_length
    with torch.no_grad():
        loss = 0
        model.eval()
        for _ in range(num_samples):
            x, y = get_batch(batch_size, val_dataset)
            logits = model.forward(x)
            loss += F.cross_entropy(logits.view(batch_size*seq_length, -1), y.view(batch_size*seq_length))
    model.train()
    return loss/num_samples

def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_dataset: TextData,
    val_dataset: TextData,
    batch_size: int,
    num_samples_train: int,
    num_samples_val: int,
    eval_every: int,
    smoothing: float = 0.9,
    verbose=True
):

    seq_length = train_dataset.seq_length
    smoothing = 0.9
    train_loss_avg = None
    for i in range(num_samples_train):
        
        x, y = get_batch(batch_size, train_dataset)
        logits = model.forward(x)
        loss = F.cross_entropy(logits.view(batch_size*seq_length, -1), y.view(batch_size*seq_length))
        if train_loss_avg is None:
            train_loss_avg = float(loss)
        else:
            train_loss_avg = smoothing*train_loss_avg + (1.0-smoothing) * float(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % eval_every == 0:
            # FIXME: should I use a fixed set for validation?
            val_loss = evaluate(model, val_dataset, batch_size, num_samples_val)
            if verbose: 
                print(
                    "i: {:6d}, train_loss: {:.2f}, val_loss: {:.2f}".
                    format(i, train_loss_avg, val_loss)
                )

if __name__ == "__main__":
    textdata = TextData("shakespeare.txt", 32, True, train_frac=0.9)
    print(f"vocabulary: {textdata.vocab}")
    x, y = textdata[0]
    print(f"x: {x}")
    print(f"y: {y}")
    print(repr(f"decode(x): {textdata.decode(x)}"))
    print(repr(f"decode(y): {textdata.decode(y)}"))
    x, y = get_batch(4, textdata)
    print(f"batch x: {x}")
    print(f"batch y: {y}")


