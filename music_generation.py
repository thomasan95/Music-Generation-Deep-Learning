import torch
from network import LSTM
import argparse
import torch.optim as optim
from torch.autograd import Variable
import string
import utilities as utils
import numpy as np


parser = argparse.ArgumentParser(description="Specify parameters for network")
parser.add_argument("-bz", "--batch_size", type=int, default=32, help="Specify batch size for network")
parser.add_argument("-nu", "--num_units", type=int, default=100, help="Specify hidden units for network")
parser.add_argument("-e", "--max_epochs", type=int, default=2000, help="Specify number of epochs to train network")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Specify learning rate of the network")
parser.add_argument("-l", "--num_layers", type=int, default=1, help="Specify number of layers for network")
parser.add_argument("-s", "--split_pct", type=float, default=0.8, help="Specify how much of training data to keep and rest for validation")
parser.add_argument("-t", "--training", type=bool, default=True, help="Specify boolean whether network is to train or to generate")
parser.add_argument("-r", "--resume", type=bool, default=False, help="Specify boolean whether to load a saved network")
args = parser.parse_args()


def train(model, train_data, valid_data, batch_size, max_epochs, criterion, optimizer, resume, char2int, update_check=100,):
    gpu = torch.cuda.is_available()
    losses = {'train': [], 'valid': []}
    min_loss = 0
    # If GPU is available, change network to run on GPU
    if gpu:
        model = model.cuda()

    if resume:
        model, optimizer, curr_epoch = utils.resume(model, optimizer)
    else:
        curr_epoch = 0

    for epoch_i in range(max_epochs - curr_epoch):
        loss = 0
        curr_loss = 0

        # Slowly increase batch_size during training
        if epoch_i % 100 == 0 and epoch_i > 0:
            batch_size = batch_size + 5

        # Tokenize the strings and convert to tensors then variables to feed into network
        batch_x, batch_y = utils.random_data_sample(train_data, batch_size)
        batch_x, batch_y = utils.string_to_tensor(batch_x, char2int), utils.string_to_tensor(batch_y, char2int)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        # Initialize model state
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


def main(batch_size, max_epochs, num_units, num_layers, lr, split_pct, training, resume):

    with open('./data/input.txt', 'r') as f:
        inp = f.read()
        char2int = utils.char_to_int(inp)
        train_data, valid_data = utils.grab_data(split_pct)

    num_outputs = len(char2int)

    # Initialize recurrent network
    model = LSTM.LSTM(batch_size, num_units, num_layers, num_outputs)

    # Initialize Loss function for network
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize optimizer for network
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Grab tokenizing dictionary, values are based off characters since we are doing character
    # by character generation
    # char2int = utils.char_to_int()

    # if training:
    model, losses = train(model, train_data, valid_data, batch_size, max_epochs, criterion, optimizer, resume, char2int)
    # else:
    #     model, optimizer, epoch = utils.resume(model, criterion, optimizer)
    #     generate_music(model)

if __name__=="__main__":
    main(args.batch_size, args.max_epochs, args.num_units, args.num_layers,
         args.lr, args.split_pct, args.training, args.resume)
