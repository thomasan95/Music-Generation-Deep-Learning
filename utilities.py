import torch
import unidecode
import random
import string
import collections

def char_to_int(text):
    '''
    :param text: pass in the entire text to get tokenized
    :param char2int: dictionary to add to
    :return: a dictionary to tokenize all printable characters
    '''
    char2int = {}
    value = 0
    for c in text:
        if c not in char2int:
            char2int[c] = value
            value += 1
    return char2int


def grab_data(split_pct):
    '''
    utility function to read in the data
    :param split_pct: amount of data to split into validation and test
    :return: training and validation sets
    '''
    assert split_pct>=0 and split_pct<=1.0
    assert isinstance(float, split_pct)

    f = './data/input.txt'
    music_data = unidecode.unidecode(open(f).read())
    n = len(music_data)
    split_idx = int(music_data*0.8)
    return music_data[:split_idx], music_data[split_idx:]


def random_data_sample(data, batch_size):
    '''
    :param data: entire data sequence to train on
    :param batch_size: batch_size
    :return: an input and target values returned for network
    '''
    assert isinstance(int,batch_size)

    n = len(data)
    random_idx = random.randint(0, n - batch_size - 1)
    x = data[random_idx:random_idx + batch_size]
    y = data[random_idx + 1:random_idx + batch_size + 1]
    return x, y


def string_to_tensor(string, dictionary):
    '''
    :param string: batch string that we would convert to tensor
    :param dictionary: char2int dictionary to tokenize the string
    :return: tensor of tokenized string
    '''
    tensor = torch.zeros(len(string)).long()
    for i, c in enumerate(string):
        tensor[i] = dictionary[c]
    return tensor


def checkpoint(state, file_name='./saves/checkpoint.pth.tar'):
    torch.save(state, file_name)


def resume(model, optimizer, filepath='./saves/checkpoint.pth.tar'):
    f = torch.load(filepath)
    epoch = f['epoch']
    model.load_state_dict(f['state_dict'])
    optimizer.load_state_dict(f['optimizer'])
    return model, optimizer, epoch
