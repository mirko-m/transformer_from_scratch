import torch
import time
from utils import TextData, train
from models import Transformer
import argparse

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

torch.manual_seed(4331)

def train_transfomer(input_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_data = TextData(input_path, SEQ_LENGTH, True, train_frac=TRAIN_FRAC)
    val_data = TextData(input_path, SEQ_LENGTH, False, train_frac=TRAIN_FRAC)
    model = Transformer(train_data.vocab_size, HIDDEN_SIZE, NUM_HEADS, NUM_BLOCKS, SEQ_LENGTH, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    tic = time.perf_counter()
    train(model, optimizer, train_data, val_data, BATCH_SIZE, NUM_SAMPLES_TRAIN, NUM_SAMPLES_VAL, EVAL_EVERY, device=device)
    print(f"Training finished in {time.perf_counter() - tic}s.")

def generate(input_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_data = TextData(input_path, SEQ_LENGTH, False, train_frac=TRAIN_FRAC)
    model = Transformer(val_data.vocab_size, HIDDEN_SIZE, NUM_HEADS, NUM_BLOCKS, SEQ_LENGTH, DROPOUT)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.eval()
    model.to(device)
    with torch.no_grad():
        x, _ = val_data[0]
        x = x.view(1, -1).to(device)
        for c in val_data.decode(x[0]):
            print(c, end="")
        print("<|>", end="")
        y = x
        for _ in range(1000):
            y = model.generate_next_token(y[:, -SEQ_LENGTH:])
            print(val_data.decode(y[0][-1:]), end="")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model or generate text using a trained model.")
    parser.add_argument("--train", action="store_true", help="Train a model.")
    parser.add_argument("--generate", action="store_true", help="Generate text using a saved model.")
    parser.add_argument("--input", action="store", type=str, help="Path to input file with text data.")
    parser.add_argument("--checkpoint", action="store", type=str, help="Path to model checkpoint for text generation.")

    args = parser.parse_args()
    if args.train and args.generate:
        raise Exception("You can only pass one of --train or --generate.")
    if not args.input:
        raise Exception("Must specify a path to the input file using --input.")
    if args.train:
        train_transfomer(args.input)
    elif args.generate:
        if not args.checkpoint:
            raise Exception("For generation a model checkpoint needs to be specified using --checkpoint.")
        generate(args.input, args.checkpoint)

