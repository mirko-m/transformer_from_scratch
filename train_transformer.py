import torch
from utils import TextData, train
from models import Transformer

if __name__ == "__main__":
    torch.manual_seed(4331)
    train_data = TextData("shakespeare.txt", 8, True, train_frac=0.9)
    val_data = TextData("shakespeare.txt", 8, False, train_frac=0.9)
    model = Transformer(train_data.vocab_size, 32, 4, 3)
    optimizer = torch.optim.Adam(model.parameters())

    train(model, optimizer, train_data, val_data, 32, 10000, 1000, 1000)
