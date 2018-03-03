import torch
import random
import pickle


def grab_data(split_pct, music_data):
    '''
     Utility function to read in the data

    :param split_pct: amount of data to split into validation and test
    :type split_pct: float
    :param music_data: The music file in ABC format in one continuous string
    :type music_data: str
    :return: training and validation sets
    :rtype: list, list
    '''

    assert 0 <= split_pct <= 1.0
    assert isinstance(split_pct, float)
    assert isinstance(music_data, str)
    
    num_files = music_data.count('<start>')
    num_split = int(num_files*split_pct)
    splits = music_data.split('<start>', num_split+1)
    split_idx = len(music_data) - len(splits[-1]) - len('<start>')
    return list(music_data[:split_idx]), list(music_data[split_idx:])


def sequential_data_sample(data, seq_len, batch_size, start_idx):
    assert isinstance(batch_size, int)
    assert isinstance(data, list)
    assert isinstance(start_idx, int)
    assert seq_len > 0
    assert batch_size > 0
    assert start_idx >= 0

    n = len(data)

    x, y = [], []
    for b in range(batch_size):
        if (start_idx+seq_len+1) >= n:
            start_idx = seq_len-(n-start_idx)

        x.append(data[start_idx:start_idx + seq_len])
        y.append(data[start_idx + 1:start_idx + seq_len + 1])
        start_idx = start_idx+seq_len

    return start_idx, x, y


def random_data_sample(data, seq_len, batch_size):
    ''' Grabs a random chunk of data from the training set.
    It will shift the target by one from the input values, so the network will be
    able to train character by character

    :param data: entire data sequence to train on
    :type data: list
    :param seq_len: how long of sequence to sample
    :type seq_len: int
    :param batch_size: batch_size
    :type batch_size: int
    :return: an input and target values returned for network
    :rtype: str, str
    '''

    assert isinstance(batch_size, int)
    assert isinstance(data, list)
    assert seq_len > 0
    assert batch_size > 0

    n = len(data)

    x, y = [], []
    for b in range(batch_size):
        random_idx = random.randint(0, n - seq_len - 1)
        x.append(data[random_idx:random_idx + seq_len])
        y.append(data[random_idx + 1:random_idx + seq_len + 1])
    return x, y


def val_to_tensor(val, dictionary, batch_size, labels=False):
    '''
    Performs padding on validation set and then reshapes into appropriate tensors
    :param val: validation set
    :type val: list
    :param dictionary: char2int to tokenize validation
    :type dictionary: dict
    :param batch_size: how much to split validation into
    :param batch_size: int
    :param labels: whether to cast tensor into long or keep as float
    :type labels: bool
    :return: tensor of validation and sequence length
    :rtype: torch.LongTensor or torch.FloatTensor, int
    '''
    amount_to_pad = len(val)%batch_size
    val += ['<pad>']*amount_to_pad
    val_seq_len = len(val)//batch_size
    if labels:
        tensor = torch.zeros(val_seq_len, batch_size, 1).long()
    else:
        tensor = torch.zeros(val_seq_len, batch_size, 1).float()

    for batch_i in range(batch_size):
        for seq_i in range(val_seq_len):
            tensor[seq_i, batch_i, 0] = dictionary[val[batch_i*val_seq_len + seq_i]]

    return tensor


def string_to_tensor(inp, dictionary, batch_size, seq_len, labels=False):
    '''
    Changes the string into tokenized tensor string

    :param inp: batch string that we would convert to tensor
    :type inp: list
    :param dictionary: char2int dictionary to tokenize the string
    :type dictionary: dict
    :param batch_size: how many sequences to grab
    :type batch_size: int
    :param seq_len: how long each sequence is
    :type seq_len: int
    :param labels: specifies whether string passed in is our target or train
    :type labels: bool
    :return: tensor of tokenized string
    :rtype: torch.FloatTensor, torch.LongTensor
    '''

    assert isinstance(inp, list)
    assert isinstance(dictionary, dict)
    assert isinstance(batch_size, int)
    assert isinstance(labels, bool)

    if labels:
        tensor = torch.zeros(seq_len, batch_size, 1).long()
    else:
        tensor = torch.zeros(seq_len, batch_size, 1).float()

    for batch_i in range(batch_size):
        for seq_i in range(seq_len):
            tensor[seq_i, batch_i, 0] = dictionary[inp[batch_i][seq_i]]

    return tensor


def checkpoint(state, file_name='./saves/checkpoint.pth.tar'):
    '''
    Save the PyTorch model

    :param state: Contains everything to be stored
    :type state: dict
    :param file_name: path where to save the file
    :type file_name: str
    '''

    assert isinstance(state, dict)
    assert isinstance(file_name, str)

    torch.save(state, file_name)


def resume(model, optimizer, gpu, filepath='./saves/checkpoint.pth.tar'):
    '''
    Loads the the saved PyTorch model at the specified location

    :param model: Initialized model
    :type model: PyTorch model
    :param optimizer: Optimizer to resume state dict
    :type optimizer: torch.optim.adam.Adam, torch.optim.rmsprop.RMSprop, torch.optim.adagrad.Adagrad
    :param filepath: location of where model is saved
    :param gpu: boolean for whether to load as gpu model or cpu model
    :type gpu: bool
    :type filepath: str
    :return: saved PyTorch model
    :rtype: PyTorch model
    '''
    assert isinstance(filepath, str)

    if gpu:
        f = torch.load(filepath)
    else:
        f = torch.load(filepath, map_location=lambda storage, loc: storage)
    epoch = f['epoch']
    losses = f['losses']
    seq_len = f['seq_len']

    model.load_state_dict(f['state_dict'])
    optimizer.load_state_dict(f['optimizer'])
    return model, optimizer, epoch, losses, seq_len


def early_stop(val_loss):
    '''
    Implement Early Stopping in the function

    :param val_loss: List of validation losses
    :type val_loss: list
    :return: bool of whether to early stop or not
    :rtype: bool
    '''

    assert isinstance(val_loss, list)

    if val_loss[-1] > val_loss[-2] > val_loss[-3] > val_loss[-4] > val_loss[-5] > val_loss[-6]:
        return True
    else:
        return False


def pickle_files(filename, stuff):
    '''
    Save files to be loaded in the future
    :param filename: name of file to save
    :type filename: str
    :param stuff: Datastructure to be saved
    :return: None
    '''
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def load_files(filename):
    '''
    Load files from pickle
    :param filename: name of pickle to load
    :type filename: str
    :return: None
    '''
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def song_parser(file_path):
    '''
    loads file of generated songs and splits up by <start> <end> terminators returns as list of strings
    :param file_path: specifies file to open
    :type file_path: str
    :return: list of strings where each string is a song
    :rtype: list(str)
    '''

    music_file = open(file_path,'r')
    all_songs = music_file.read()
    list_songs = []
    loc = 0
    found = 0
    while(found != -1):
        loc = all_songs.find('<start>',loc)
        found = all_songs.find('<start>',loc+1)
        if found == -1:
            found = all_songs.find('<end>',loc+1)
            if found != -1:
                list_songs.append(all_songs[loc:found+len('<end>')])
                return list_songs
            else: 
                return list_songs
        else:
            list_songs.append(all_songs[loc:found+len('<end>')])
            loc = found
