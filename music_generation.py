import torch
from data.network import LSTM
import argparse
import torch.optim as optim


def train(batch_size, hidden_units, num_layers, num_outputs, max_epochs, lr=0.001):
    gpu = torch.cuda.is_available()
    lstm = LSTM.LSTM(batch_size, hidden_units, num_layers, num_outputs)

    # If GPU is available, change network to run on GPU
    if gpu:
        lstm = lstm.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    for epoch_i in range(max_epochs):
        lstm.zero_grad()
        lstm.hidden = lstm.init_hidden()


def main(batch_size, max_epochs, num_units):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bz", "--batch_size", help="specify batch size for network")
    parser.add_argument("-h", "--num_units", help="specify hidden units for network")
    parser.add_argument("-e", "--max_epochs", help="specify number of epochs to train network")
    parser.add_argument("-lr", "--learning_rate", help="specify learning rate of the network")
    args = parser.parse_args()

    # Batch Size
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 32

    # Max Epochs
    if args.max_epochs:
        max_epochs = int(args.max_epochs)
    else:
        max_epochs = 100

    if args.num_units:
        num_units = int(args.num_units)
    else:
        num_units = 100

    main(batch_size, max_epochs, num_units)