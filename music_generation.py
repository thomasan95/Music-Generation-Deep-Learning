import torch
from network import LSTM
import argparse
import torch.optim as optim
import string
import utilities as utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-bz", "--batch_size", help="Specify batch size for network")
parser.add_argument("-h", "--num_units", help="Specify hidden units for network")
parser.add_argument("-e", "--max_epochs", help="Specify number of epochs to train network")
parser.add_argument("-lr", "--learning_rate", help="Specify learning rate of the network")
parser.add_argument("-l", "--num_layers", help="Specify number of layers for network")
parser.add_argument("-s", "--split_pct", help="Specify how much of training data to keep and rest for validation")
parser.add_argument("-t", "--training", help="Specify boolean whether network is to train or to generate")
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
    max_epochs = 2000

if args.num_units:
    num_units = int(args.num_units)
else:
    num_units = 100

if args.num_layers:
    num_layers = int(num_layers)
else:
    num_layers = 1

if args.learning_rate:
    lr = float(args.learning_rate)
else:
    lr = 0.001

if args.split_pct:
    split_pct = float(args.split_pct)
else:
    split_pct = 0.8

if args.resume:
    resume = True
else:
    resume = False

if args.training:
    training = args.training
else:
    training = False


def train(model, train_data, valid_data, batch_size, max_epochs, criterion, optimizer, update_check=100):
    gpu = torch.cuda.is_available()
    losses = {'train': [], 'valid': []}
    min_loss = 0
    # If GPU is available, change network to run on GPU
    if gpu:
        model = model.cuda()

    for epoch_i in range(max_epochs):
        loss = 0
        curr_loss = 0

        # Slowly increase batch_size during training
        if epoch_i % 100 == 0 and epoch_i > 0:
            batch_size = batch_size + 5

        batch_x, batch_y = utils.random_data_sample(train_data, batch_size)
        model.hidden = model.init_hidden()
        for index in range(len(batch_x)):
            model.zero_grad()
            output = model(batch_x[index])
            loss += criterion(output, batch_y[index])
            curr_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        curr_loss = curr_loss/len(batch_x)
        losses['train'].append(curr_loss)

        if epoch_i == 0:
            min_loss = losses['train'][-1]

        if epoch_i%update_check == 0 and epoch_i > 0:
            update_loss = np.mean(losses['train'][-100:])
            if update_loss < min_loss:
                print("New Best Model! Saving!")
                min_loss = update_loss
                utils.checkpoint({'epoch': epoch_i,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()})

    return model, losses


def generate_music(model):
    pass # To Do


def main(batch_size, max_epochs, num_units, num_layers, lr, split_pct, training):
    characters = string.printable
    num_outputs = len(characters)

    train_data, valid_data = utils.grab_data(split_pct)

    # Initialize recurrent network
    model = LSTM.LSTM(batch_size, num_units, num_layers, num_outputs)

    # Initialize Loss function for network
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize optimizer for network
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if training:
        model, losses = train(model, train_data, valid_data, batch_size, max_epochs, criterion, optimizer)
    else:
        model, optimizer, epoch = utils.resume(model, criterion, optimizer)
        generate_music(model)

if __name__=="__main__":
    global batch_size
    global max_epochs
    global num_units
    global num_layers
    global lr
    global split_pct
    global training
    main(batch_size, max_epochs, num_units, num_layers, lr, split_pct, training)