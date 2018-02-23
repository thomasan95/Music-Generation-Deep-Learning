import torch
import random


# def char_to_int(text):
#     '''
#     :param text: pass in the entire text to get tokenized
#     :param char2int: dictionary to add to
#     :return: a dictionary to tokenize all ABC characters
#     '''
#
#     assert isinstance(text, str)
#
#     chars = list(set(text))
#     char2int = {}
#     for i, c in enumerate(chars):
#         char2int[c] = i
#     return char2int


def grab_data(split_pct, music_data):
    '''
    utility function to read in the data
    :param split_pct: amount of data to split into validation and test
    :return: training and validation sets
    '''
    assert 0 <= split_pct <= 1.0
    assert isinstance(split_pct, float)

    n = len(music_data)
    split_idx = int(n*0.8)
    return music_data[:split_idx], music_data[split_idx:]


def random_data_sample(data, batch_size):
    '''
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
    if val_loss[-1] > val_loss[-2] and val_loss[-2] > val_loss[-3]:
        return True
    else:
        return False
def reverse_lookup_dict(dictionary):
    '''
    Create reverse lookup dictionary for converting prediction numbers to characters
    :param dictionary: tokenizing dictionary
    :return: reverse lookup dictionary
    '''
    lookup = {}
    for a, b in dictionary.items():
        lookup[b] = a
    return lookup
