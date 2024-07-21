import torch
from models import Transformer
from utils import TextData

if __name__ == "__main__":
    
    torch.manual_seed(4331)
    val_data = TextData("shakespeare.txt", 256, False, train_frac=0.9)
    model = Transformer(val_data.vocab_size, 384, 6, 6)
    model.load_state_dict(torch.load("./model_checkpoint.pt"))
    model = model.eval()
    with torch.no_grad():
        x, _ = val_data[0]
        x = x.view(1, -1)
        for c in val_data.decode(x[0]):
            print(c, end="")
        print("<|>", end="")
        y = x
        for i in range(1000):
            y = model.generate(y[:, -256:], 1)
            print(val_data.decode(y[0][-1:]), end="")
        print("\n")
    