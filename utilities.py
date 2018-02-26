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


def resume(model, filepath='./saves/checkpoint.pth.tar'):
    '''
    Loads the the saved PyTorch model at the specified location

    :param model: Initialized model
    :type model: PyTorch model
    :param filepath: location of where model is saved
    :type filepath: str
    :return: saved PyTorch model
    :rtype: PyTorch model
    '''
    assert isinstance(filepath, str)

    f = torch.load(filepath)
    # epoch = f['epoch']
    model.load_state_dict(f['state_dict'])
    # optimizer.load_state_dict(f['optimizer'])
    return model


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
