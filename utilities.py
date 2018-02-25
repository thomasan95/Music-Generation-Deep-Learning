import torch
import random


def grab_data(split_pct, music_data):

    ''' Utility function to read in the data

    :param split_pct: amount of data to split into validation and test
    :type split_pct: float
    :param music_data: The music file in ABC format in one continuous string
    :type music_data: str
    :return: training and validation sets
    '''

    assert 0 <= split_pct <= 1.0
    assert isinstance(split_pct, float)

#    n = len(music_data)
#    split_idx = int(n*0.8)
    
    num_files = music_data.count('<start>')
    split_idx = 0
    for f in range(int(num_files*0.8)):
        split_idx = music_data.find('<start>',split_idx)
    return music_data[:split_idx], music_data[split_idx:]


def random_data_sample(data, batch_size):
    ''' Grabs a random chunk of data from the training set.
    It will shift the target by one from the input values, so the network will be
    able to train character by character

    :param data: entire data sequence to train on
    :type data: str
    :param batch_size: batch_size
    :type batch_size: int
    :return: an input and target values returned for network
    :rtype: str, str
    '''
    assert isinstance(batch_size, int)
    assert isinstance(data, str)
    assert batch_size > 0

    n = len(data)
    random_idx = random.randint(0, n - batch_size - 1)
    x = data[random_idx:random_idx + batch_size]
    y = data[random_idx + 1:random_idx + batch_size + 1]
    return x, y


def string_to_tensor(string, dictionary, labels=False):
    '''
    Changes the string into tokenized tensor string

    :param string: batch string that we would convert to tensor
    :type string: str
    :param dictionary: char2int dictionary to tokenize the string
    :type dictionary: dict
    :return: tensor of tokenized string
    :rtype: torch tensor
    '''

    assert isinstance(string, str)
    assert isinstance(dictionary, dict)

    if labels:
        tensor = torch.zeros(len(string)).long()
    else:
        tensor = torch.zeros(len(string)).float()
    for i, c in enumerate(string):
        tensor[i] = dictionary[c]
    return tensor


def checkpoint(state, file_name='./saves/checkpoint.pth.tar'):
    '''
    Save the PyTorch model

    :param state: Contains everything to be stored
    :type state: dict
    :param file_name: path where to save the file
    :type file_name: str
    '''
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

    if val_loss[-1] > val_loss[-2] > val_loss[-3] > val_loss[-4]:
        return True
    else:
        return False
