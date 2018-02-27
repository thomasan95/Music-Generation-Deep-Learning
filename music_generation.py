import torch
from network import LSTM, GRU
import argparse
import torch.optim as optim
from torch.autograd import Variable
import utilities as utils
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import datetime


parser = argparse.ArgumentParser(description="Specify parameters for network")
parser.add_argument("-bz", "--batch_size", type=int, default=1, help="Specify batch size for network")
parser.add_argument("-sl", "--seq_len", type=int, default=5, help="Initial sequence length to train on")
parser.add_argument("-nu", "--num_units", type=int, default=100, help="Specify hidden units for network")
parser.add_argument("-e", "--max_epochs", type=int, default=1000000, help="Specify number of epochs to train network")
parser.add_argument("-thresh", "--threshold", type=float, default=2.5, help="Threshold for when to increase batch_size")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Specify learning rate of the network")
parser.add_argument("-l", "--num_layers", type=int, default=1, help="Specify number of layers for network")
parser.add_argument("-s", "--split_pct", type=float, default=0.8,
                    help="Specify how much of training data to keep and rest for validation")
parser.add_argument("-t", "--training", type=str, default='true',
                    help="Specify boolean whether network is to train or to generate")
parser.add_argument("-d", "--dropout", type=float, default=0, help="Specify amount of dropout after each layer in LSTM")
parser.add_argument("-n", "--network", type=str, default='LSTM', help="Specify whether use GRU or LSTM")
parser.add_argument("-uc", "--update_check", type=int, default=1000, help="How often to check save model")
parser.add_argument("-f", "--file", type=str, default='./data/input.txt', help="Input file to train on")
parser.add_argument("-gf", "--generate_file", type=str, default='./generate/gen.txt',
                    help="Path to save generated file")
parser.add_argument("-gc", "--generate_length", type=int, default=5000, help="How many characters to generate")
parser.add_argument("-temp", "--temperature", type=float, default=0.8, help="Temperature for network")
parser.add_argument("--save_append", type=str, default="('{:%b_%d_%H:%M}'.format(datetime.datetime.now()))", 
                    help="What to append to save path to make it unique")
parser.add_argument("-rt", "--resume_training", type=str, default='False', help="Specify whether to continue training a saved model")
parser.add_argument("-es", "--early_stop", type=str, default='true', help="Specify whether to use early stopping")
parser.add_argument("-ms", "--max_seq_len", type=int, default=700, help="max length of input to batch")
parser.add_argument("-op", "--optim", type=str, default='Adam', help="Specify type of optimizer for network")
parser.add_argument("-un", "--unit_number", type=int, default=0, help="the unit number that you wish to generate a heat map for")
parser.add_argument("-ghm", "--generate_heat_map", type=str, default='false', help="whether you wish to generate songs and then a heat map")
parser.add_argument("-hm", "--heat_map", type=str, default='false', help="whether you wish to generate a heat map for pregenerated song")
parser.add_argument("-hmp", "--heat_map_path", type=str, default='false', help="path of the pregenerated song you want to generate a heat map for")
parser.add_argument("-us", "--update_seq", type=str, default='valid', help="Update sequence length based off of validation loss or train loss")

args = parser.parse_args()

gpu = torch.cuda.is_available()
if gpu:
    print("\nRunning on GPU\n")


def train(model, train_data, valid_data, seq_len, criterion, optimizer, char2int, losses = {'train': [], 'valid': []}, epoch=0):
    '''
    Function trains the model. It will save the current model every update_check iterations so model can then be
    loaded and resumed either for training or for music generation in the future

    :param model: Recurrent network model to be passed in
    :type model: PyTorch model
    :param train_data: data to be passed in to be considered as part of training
    :type train_data: list
    :param valid_data: data to be passed in to be considered as part of validation
    :type valid_data: list
    :param seq_len: initial seq_len to start with for training
    :type seq_len: int
    :param criterion: Loss function to be used (CrossEntropyLoss)
    :type criterion: PyTorch Loss
    :param optimizer: PyTorch optimizer to user in the training process (Adam or SGD or RMSProp)
    :type optimizer: PyTorch Optimizer
    :param char2int: type dict, Dictionary to tokenize the batches
    :type char2int: dict
    :param losses: dictionary containing the training and validation loss
    :type losses: dict
    :return: The trained model and the corresponding losses
    :rtype: PyTorch Model, dict
    '''
    
    avg_val_loss = 0
    running_mean_benchmark = 3.0
    valid_x, valid_y = valid_data[:-1], valid_data[1:]
    valid_x, val_seq_len = utils.val_to_tensor(valid_x, char2int, args.batch_size)
    valid_y, _ = utils.val_to_tensor(valid_y, char2int, args.batch_size, labels=True)
    valid_x, valid_y = Variable(valid_x), Variable(valid_y)

    # If GPU is available, change network to run on GPU
    if gpu:
        model = model.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
    times = []
    temp_loss = []

    for epoch_i in range(epoch, args.max_epochs + 1):
        loss = 0

        # Slowly increase seq_len during training
        # if epoch_i > 100 and epoch_i % 100 == 0:

        # if epoch_i % 1000 == 0:
        #     if seq_len < args.max_seq_len:
        #         seq_len += 5
        #         print("\nIncreasing sequence length to: " + str(seq_len))

        # Tokenize the strings and convert to tensors then variables to feed into network
        batch_x, batch_y = utils.random_data_sample(train_data, seq_len, args.batch_size)
        batch_x = utils.string_to_tensor(batch_x, char2int, args.batch_size, seq_len)
        batch_y = utils.string_to_tensor(batch_y, char2int, args.batch_size, seq_len, labels=True)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        if gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        # Initialize model state
        model.hidden = model.init_hidden()
        start = timer()
        for index in range(len(batch_x)):
            model.zero_grad()
            output = model(batch_x[index, :, 0])
            loss += criterion(torch.squeeze(output, dim=0), batch_y[index, :, 0])

        delta_time = timer() - start

        times.append(delta_time)
        curr_loss = loss.data[0]
        loss.backward()
        optimizer.step()
        curr_loss = curr_loss / len(batch_x)
        temp_loss.append(curr_loss)

        if epoch_i % args.update_check == 0 and epoch_i > 0:
            valid_loss = 0
            for i in range(val_seq_len):
                valid_out = model(valid_x[i, :, 0])
                valid_loss += criterion(torch.squeeze(valid_out, dim=0), valid_y[i, :, 0])

            avg_val_loss = valid_loss.data[0] / val_seq_len
            losses['valid'].append(avg_val_loss)
            losses['train'].append(sum(temp_loss)/len(temp_loss))
            temp_loss = []
            utils.pickle_files('./results/losses-' + str(args.save_append) + '.p', losses)


            if losses['valid'][-1] <= min(losses['valid']):
                print("New Best Model! Saving!")
                utils.checkpoint({'epoch': epoch_i, 
                                  'losses': losses, 
                                  'seq_len': seq_len,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()},
                                  './saves/checkpoint-' + str(args.save_append) + '.pth.tar')

            if len(losses['valid']) > 4 and args.early_stop == 'true':
                early_stop = utils.early_stop(losses['valid'])
                if early_stop:
                    print("Stopping due to Early Stop Criterion")
                    break

            # Update sequence length
            if args.update_seq == 'valid':
                if seq_len < args.max_seq_len and len(losses['valid']) > 1 and (losses['valid'][-1] < losses['valid'][-2]):
                    seq_len += int(2/(losses['valid'][-2] - losses['valid'][-1]))
                    if seq_len > args.max_seq_len:
                        seq_len = args.max_seq_len
                    print("\nIncreasing sequence length to: " + str(seq_len))
            elif args.update_seq == 'train':
                if seq_len < args.max_seq_len and (sum(losses['train'])/len(losses['train'])) < running_mean_benchmark:
                    delta_rmean = round(running_mean_benchmark - sum(losses['train'])/len(losses['train']), 1)
                    seq_len = int(seq_len * 1.5**(delta_rmean*10))
                    if seq_len > args.max_seq_len:
                        seq_len = args.max_seq_len
                    running_mean_benchmark -= delta_rmean
                    print("\nIncreasing sequence length to: " + str(seq_len))

        if epoch_i % 100 == 0 and epoch_i > 0:
            times = np.asarray(times)

            # losses['train'].append(sum(temp_loss)/len(temp_loss))
            # temp_loss = []
            #
            # if losses['train'][-1] < args.threshold:
            #     seq_len += 2

            print("Epoch: %d\tCurrent Train Loss:%f\tValid Loss (since last check):%f\tAvg Time Per Batch:%f" %
                  (epoch_i, curr_loss, avg_val_loss, np.mean(times).astype(float)))
            times = []

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
        primer_input = [list(primer[:-1])]
        primer_input = Variable(utils.string_to_tensor(primer_input, char2int, batch_size=1, seq_len=len(primer_input)))
        if gpu:
            primer_input = primer_input.cuda()

        for i in range(len(primer_input)):
            _ = model(primer_input[i])

        inp = Variable(utils.string_to_tensor(list(primer[-1]), char2int, batch_size=1, seq_len=1))

        if gpu:
            inp = inp.cuda()

        for c_idx in range(args.generate_length):
            out = model(inp)
            # Normalize distribution by temperature and turn into a vector
            out = out.data.view(-1).div(args.temperature).exp()
            out = out.div(out.sum())
            # Sample num_samples from multinomial distribution
            next_input = torch.multinomial(out, num_samples)[0]
            predict_char = int2char[next_input]
            return_string += predict_char
            predict_char = list(predict_char)
            inp = Variable(utils.string_to_tensor(predict_char, char2int, batch_size=1, seq_len=1))
            if gpu:
                inp = inp.cuda()
        f.write(return_string)
        f.close()


def heat_map(model, char2int, int2char, unit_num=0, song_path='saves/song.txt', file_path='saves/heatmap.png'):
    load_song = open(song_path, 'r')
    generated_song = list(load_song.read())
    
    tensor_song = utils.string_to_tensor([generated_song], char2int , 1, len(generated_song))
    tensor_song = Variable(tensor_song)
    if gpu:
        tensor_song = tensor_song.cuda()
    activations = []
    for index in range(len(tensor_song)):
        model.zero_grad()
        output = model(tensor_song[index])
        hidden, cell = model.hidden
        cell = cell.data.numpy()
        hidden = hidden.data.numpy()
        # print cell[0,0,0]
        activations.append(hidden[0, 0, unit_num])
    activations = np.asarray(activations)
    song_length = activations.shape[0]
    height = int(np.sqrt(song_length))
    width = song_length / height + 1
    song_activations = np.zeros(height * width)
    song_activations[:song_length] = activations[:]
    song_activations = np.reshape(song_activations, (height, width))
    song_activations = [x for x in song_activations[::-1]]
    plt.figure()
    heatmap = plt.pcolormesh(song_activations,cmap = 'coolwarm')

    countH = height-1
    countW = 0
    for index in range(len(generated_song)):
        plt.text(countW, countH, generated_song[index])
        countW += 1
        if countW >= width:
            countH -= 1
            countW = 0

    plt.colorbar(heatmap)
    plt.show()
    return song_activations


def main():
    with open(args.file, 'r') as f:
        inp = f.read()
        # Create tokenizing dictionary for text in ABC notation
        char2int = dict((a, b) for b, a in enumerate(list(set(inp))))
        char2int['<pad>'] = len(char2int)
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
    if args.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    elif args.optim == "RMS":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.training == 'true' and args.resume_training=='True':
        print('Loading model...')
        model, optimizer, epoch_i, losses, seq_len = utils.resume(model, optimizer, filepath=('./saves/checkpoint-' + str(args.save_append) + '.pth.tar'))
        print('Resuming training with model loaded from ' + './saves/checkpoint-' + str(args.save_append) + '.pth.tar')
        print('Epoch: %d\tCurrent Train Loss: %f\tCurrent Valid Loss: %f' %(epoch_i,losses['train'][-1],losses['valid'][-1]))
        _, _ = train(model, train_data, valid_data, seq_len, criterion, optimizer, char2int, losses=losses, epoch=(epoch_i+1))

    elif args.training =='true':
        _, _ = train(model, train_data, valid_data, args.seq_len, criterion, optimizer, char2int)
    else:
        model, _, _, _, _ = utils.resume(model, optimizer, filepath=('./saves/checkpoint-' + str(args.save_append) + '.pth.tar'))
        generate_music(model, char2int, int2char)
#    hmap = heat_map(model,char2int,int2char)  

if __name__ == "__main__":
    main()
