import torch
from network import LSTM, GRU
import argparse
import torch.optim as optim
from torch.autograd import Variable
import utilities as utils
import numpy as np
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description="Specify parameters for network")
parser.add_argument("-bz", "--batch_size", type=int, default=1, help="Specify batch size for network")
parser.add_argument("-sl", "--seq_len", type=int, default=25, help="Initial sequence length to train on")
parser.add_argument("-nu", "--num_units", type=int, default=100, help="Specify hidden units for network")
parser.add_argument("-e", "--max_epochs", type=int, default=1000000, help="Specify number of epochs to train network")
parser.add_argument("-thresh", "--threshold", type=float, default=3, help="Threshold for when to increase batch_size")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Specify learning rate of the network")
parser.add_argument("-l", "--num_layers", type=int, default=1, help="Specify number of layers for network")
parser.add_argument("-s", "--split_pct", type=float, default=0.8,
                    help="Specify how much of training data to keep and rest for validation")
parser.add_argument("-t", "--training", type=str, default='true',
                    help="Specify boolean whether network is to train or to generate")
parser.add_argument("-d", "--dropout", type=float, default=0, help="Specify amount of dropout after each layer in LSTM")
parser.add_argument("-n", "--network", type=str, default='LSTM', help="Specify whether use GRU or LSTM")
parser.add_argument("-uc", "--update_check", type=int, default=5000, help="How often to check save model")
parser.add_argument("-f", "--file", type=str, default='./data/input.txt', help="Input file to train on")
parser.add_argument("-gf", "--generate_file", type=str, default='./generate/gen.txt', help="Path to save generated file")
parser.add_argument("-gc", "--generate_length", type=int, default=5000, help="How many characters to generate")
parser.add_argument("-temp", "--temperature", type=float, default=0.8, help="Temperature for network")
parser.add_argument("--save_append", type=str, default="", help="What to append to save path to make it unique")
parser.add_argument("-rf", "--resume_file", type=str, default='./saves/checkpoint.pth.tar', help="Path to file to load")
parser.add_argument("-es", "--early_stop", type=str, default='true', help="Specify whether to use early stopping")

args = parser.parse_args()

gpu = torch.cuda.is_available()


def train(model, train_data, valid_data, seq_len, criterion, optimizer, char2int):
    '''
    Function trains the model. It will save the current model every update_check iterations so model can then be
    loaded and resumed either for training or for music generation in the future

    :param model: Recurrent network model to be passed in
    :type model: PyTorch model
    :param train_data: type str, data to be passed in to be considered as part of training
    :type train_data: str
    :param valid_data: type str, data to be passed in to be considered as part of validation
    :type valid_data: str
    :param seq_len: initial seq_len to start with for training
    :type seq_len: int
    :param criterion: Loss function to be used (CrossEntropyLoss)
    :type criterion: PyTorch Loss
    :param optimizer: PyTorch optimizer to user in the training process (Adam or SGD or RMSProp)
    :type optimizer: PyTorch Optimizer
    :param char2int: type dict, Dictionary to tokenize the batches
    :type char2int: dict
    :return: The trained model and the corresponding losses
    :rtype: PyTorch Model, dict
    '''

    losses = {'train': [], 'valid': []}
    avg_val_loss = 0
    min_loss = 0

    valid_x, valid_y = valid_data[:-1], valid_data[1:]
    valid_x, valid_y = utils.string_to_tensor(valid_x, char2int), utils.string_to_tensor(valid_y, char2int, labels=True)
    valid_x, valid_y = Variable(valid_x), Variable(valid_y)

    # If GPU is available, change network to run on GPU
    if gpu:
        model = model.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
    times = []

    for epoch_i in range(1, args.max_epochs+1):
        loss = 0
        # Slowly increase seq_len during training

        if epoch_i % 100 == 0:
            if np.mean(losses['train'][-100:]) < args.threshold:
                if gpu:
                    seq_len = seq_len + 50
                else:
                    seq_len = seq_len + 5
                print("Sequence Length changed to: " + str(seq_len))

        # Tokenize the strings and convert to tensors then variables to feed into network
        batch_x, batch_y = utils.random_data_sample(train_data, seq_len,args.batch_size)
        batch_x = utils.string_to_tensor(batch_x, char2int)
        batch_y = utils.string_to_tensor(batch_y, char2int, labels=True)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        trainLoss = []
        validationLoss = []
        lossEpoch = np.zeros(2)
        if gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        # Initialize model state
        model.hidden = model.init_hidden()
        start = timer()
        for index in range(len(batch_x)):
            model.zero_grad()
            output = model(batch_x[index])
            loss += criterion(torch.squeeze(output, dim=1), batch_y[index])
        delta_time = timer() - start

        times.append(delta_time)
        curr_loss = loss.data[0]
        loss.backward()
        optimizer.step()
        curr_loss = curr_loss/len(batch_x)
        losses['train'].append(curr_loss)

        if epoch_i % 1000 == 0 and epoch_i > 0:
            valid_loss = 0
            for i in range(len(valid_x)):
                valid_out = model(valid_x[i])
                valid_loss += criterion(torch.squeeze(valid_out, dim=1), valid_y[i])

            avg_val_loss = valid_loss.data[0]/len(valid_x)
            lossEpoch[:]=avg_val_loss,epoch_i
            validationLoss.append(lossEpoch)
            np.save('saves/validLoss.npy',np.asarray(validationLoss))
            
            losses['valid'].append(avg_val_loss)
            if len(losses['valid']) > 4 and args.early_stop == 'true':
                early_stop = utils.early_stop(losses['valid'])
                if early_stop:
                    print("Stopping due to Early Stop Criterion")
                    break

        if epoch_i == 1:
            min_loss = losses['train'][-1]

        if epoch_i % 100 == 0 and epoch_i > 0:
            times = np.asarray(times)
            print("Epoch: %d\tCurrent Train Loss:%f\tValid Loss (since last check):%f\tAvg Time Per Batch:%f" %
                  (epoch_i, curr_loss, avg_val_loss, np.mean(times).astype(float)))
            times =  []
            lossEpoch[:]=curr_loss,epoch_i
            trainLoss.append(lossEpoch)
            np.save('saves/trainLoss.npy',np.asarray(trainLoss))
        if epoch_i % args.update_check == 0 and epoch_i > 0:
            update_loss = np.mean(losses['train'][-args.update_check:])
            if update_loss < min_loss:
                print("New Best Model! Saving!")
                min_loss = update_loss
                utils.checkpoint({'epoch': epoch_i,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()},
                                 './saves/checkpoint-'+str(args.network)+'-'+str(args.save_append)+'.pth.tar')
    return model, losses


def generate_music(model, char2int, int2char, file=args.generate_file, num_samples=1):
    '''
    Generate music will be called when args.training is set to 'false'. In that case, the function will generate
    a certain amount of characters specified by args.generate_length.

    :param model: Loaded model from main() to be passed into network
    :type model: PyTorch Model
    :param char2int: Dictionary to convert characters to integers
    :type char2int: dict
    :param int2char: type dict, Dictionary to convert integers to characters
    :type int2char: dict
    :param file: File path to save the generated music to
    :type file: str
    :param num_samples:  Number of samples to draw
    :type num_samples: int
    '''
    if gpu:
        model = model.cuda()

    primer = input("Please enter text to prime network: ")
    return_string = primer

    with open(file, 'w') as f:
        primer_input = primer[:-1]
        primer_input = Variable(utils.string_to_tensor(primer_input, char2int))
        if gpu:
            primer_input = primer_input.cuda()

        for i in range(len(primer_input)):
            _ = model(primer_input[i])

        inp = Variable(utils.string_to_tensor(primer[-1], char2int))

        for c_idx in range(args.generate_length):
            out = model(inp)
            # Normalize distribution by temperature and turn into a vector
            out = out.data.view(-1).div(args.temperature).exp()
            out = out.div(out.sum())
            # Sample num_samples from multinomial distribution
            next_input = torch.multinomial(out, num_samples)[0]
            predict_char = int2char[next_input]
            return_string += predict_char
            inp = Variable(utils.string_to_tensor(predict_char, char2int))
            if gpu:
                inp = inp.cuda()
        f.write(return_string)
        f.close()


def main():
    with open(args.file, 'r') as f:
        inp = f.read()
        # Create tokenizing dictionary for text in ABC notation
        char2int = dict((a,b) for b,a in enumerate(list(set(inp))))
        # Create reverse lookup dictionary for the text
        int2char = {v: k for k, v in char2int.items()}

        train_data, valid_data = utils.grab_data(args.split_pct, inp)

    # Number of total unique characters
    num_outputs = len(char2int)

    # Initialize recurrent network
    if args.network == 'GRU':
        print("Using GRU Network")
        model = GRU.GRU(args.batch_size, args.num_units, args.num_layers, num_outputs, args.dropout)
    else:
        print("Using LSTM Network")
        model = LSTM.LSTM(args.batch_size, args.num_units, args.num_layers, num_outputs, args.dropout)

    # Initialize Loss function for network
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize optimizer for network
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.training == 'true':
        _, _ = train(model, train_data, valid_data, args.seq_len, criterion, optimizer, char2int)
    else:
        model = utils.resume(model, filepath=args.resume_file)
        generate_music(model, char2int, int2char)


if __name__=="__main__":
    main()
