import torch
import time
from utils import TextData, train
from models import Transformer

# Model hyperparameters
SEQ_LENGTH = 256
BATCH_SIZE =  64
HIDDEN_SIZE = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
DROPOUT = 0.2

# Training hyperparameters
TRAIN_FRAC = 0.9
NUM_SAMPLES_TRAIN = 5000
NUM_SAMPLES_VAL = 200
EVAL_EVERY = 500

if __name__ == "__main__":
    torch.manual_seed(4331)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_data = TextData("shakespeare.txt", SEQ_LENGTH, True, train_frac=TRAIN_FRAC)
    val_data = TextData("shakespeare.txt", SEQ_LENGTH, False, train_frac=TRAIN_FRAC)
    model = Transformer(train_data.vocab_size, 384, 6, 6, SEQ_LENGTH, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    tic = time.perf_counter()
    train(model, optimizer, train_data, val_data, BATCH_SIZE, NUM_SAMPLES_TRAIN, NUM_SAMPLES_VAL, EVAL_EVERY, device=device)
    print(f"Training finished in {time.perf_counter() - tic}s.")

