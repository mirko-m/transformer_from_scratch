import torch
from utils import TextData, train
from bigram import BigramModel

if __name__ == "__main__":
    torch.manual_seed(4331)
    train_data = TextData("shakespeare.txt", 8, True, train_frac=0.9)
    val_data = TextData("shakespeare.txt", 8, False, train_frac=0.9)
    model = BigramModel(train_data.vocab_size)
    optimizer = torch.optim.Adam(model.parameters())

    train(model, optimizer, train_data, val_data, 32, 10000, 1000, 1000)
