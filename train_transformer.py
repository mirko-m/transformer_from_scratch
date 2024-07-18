import torch
import time
from utils import TextData, train
from models import Transformer, CONTEXT_LENGTH

if __name__ == "__main__":
    torch.manual_seed(4331)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    seq_length = 256
    batch_size =  64
    train_data = TextData("shakespeare.txt", seq_length, True, train_frac=0.9)
    val_data = TextData("shakespeare.txt", seq_length, False, train_frac=0.9)
    model = Transformer(train_data.vocab_size, 384, 6, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    tic = time.perf_counter()
    train(model, optimizer, train_data, val_data, batch_size, 5000, 200, 500, device=device)
    print(f"Training finished in {time.perf_counter() - tic}s.")

    for i in range(10):
        x, _ = val_data[i*seq_length]
        x = x.view(1,-1).to(device)
        # We can generate one additional character beyond CONTEXT_LENGTH
        y = model.generate(x, CONTEXT_LENGTH-seq_length+1)
        print(val_data.decode(y[0]))
        print("\n#############\n")
    
