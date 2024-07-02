import torch
from utils import TextData, train
from models import Transformer, CONTEXT_LENGTH

if __name__ == "__main__":
    torch.manual_seed(4331)
    seq_length = 8
    batch_size =  32
    train_data = TextData("shakespeare.txt", seq_length, True, train_frac=0.9)
    val_data = TextData("shakespeare.txt", seq_length, False, train_frac=0.9)
    model = Transformer(train_data.vocab_size, 32, 4, 3)
    optimizer = torch.optim.Adam(model.parameters())

    train(model, optimizer, train_data, val_data, batch_size, 10000, 1000, 1000)

    for i in range(10):
        x, _ = val_data[i*seq_length]
        x = x.view(1,-1)
        # We can generate one additional character beyond CONTEXT_LENGTH
        y = model.generate(x, CONTEXT_LENGTH-seq_length+1)
        print(val_data.decode(y[0]))
        print("\n#############\n")
