import torch
from network import LSTM
import argparse
import torch.optim as optim
import string
import utilities as utils

def train(data, batch_size, hidden_units, num_layers, num_outputs, max_epochs, lr=0.001):
    gpu = torch.cuda.is_available()
    lstm = LSTM.LSTM(batch_size, hidden_units, num_layers, num_outputs)
    losses = {'train': [], 'valid': []}
    accuracies = {'train': [], 'valid': []}

    # If GPU is available, change network to run on GPU
    if gpu:
        lstm = lstm.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    for epoch_i in range(max_epochs):
        loss = 0
        curr_loss = 0
        batch_x, batch_y = utils.random_data_sample(data, batch_size)
        lstm.hidden = lstm.init_hidden()
        for index in range(len(batch_x)):
            lstm.zero_grad()
            output = lstm(batch_x[index])
            loss += criterion(output, batch_y[index])
            curr_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        curr_loss = curr_loss/len(batch_x)
        losses['train'].append(curr_loss)




def main(batch_size, max_epochs, num_units):
    characters = string.printable
    


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