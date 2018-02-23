import torch
import random


def grab_data(split_pct, music_data):
    ''' Utility function to read in the data
    :param split_pct: amount of data to split into validation and test
    :param music_data: The music file in ABC format in one continuous string
    :return: training and validation sets
    '''
    assert 0 <= split_pct <= 1.0
    assert isinstance(split_pct, float)

    n = len(music_data)
    split_idx = int(n*0.8)
    return music_data[:split_idx], music_data[split_idx:]


def random_data_sample(data, batch_size):
    ''' Grabs a random chunk of data from the training set.
    It will shift the target by one from the input values, so the network will be
    able to train character by character

    :param data: entire data sequence to train on
    :param batch_size: batch_size
    :return: an input and target values returned for network
    '''
    assert isinstance(batch_size, int)

    n = len(data)
    random_idx = random.randint(0, n - batch_size - 1)
    x = data[random_idx:random_idx + batch_size]
    y = data[random_idx + 1:random_idx + batch_size + 1]
    return x, y


def string_to_tensor(string, dictionary, labels=False):
    '''
    :param string: batch string that we would convert to tensor
    :param dictionary: char2int dictionary to tokenize the string
    :return: tensor of tokenized string
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
    torch.save(state, file_name)


def resume(model, optimizer, filepath='./saves/checkpoint.pth.tar'):
    f = torch.load(filepath)
    epoch = f['epoch']
    model.load_state_dict(f['state_dict'])
    optimizer.load_state_dict(f['optimizer'])
    return model, optimizer, epoch

def get_accuracy(preds, labels):
    pass


def early_stop(val_loss):
    '''
    Implement Early Stopping in the function
    :param val_loss: list of validation losses
    :return: bool of whether to early stop or not
    '''
    assert isinstance(val_loss, list)

    if val_loss[-1] > val_loss[-2] and val_loss[-2] > val_loss[-3]:
        return True
    else:
        return False
