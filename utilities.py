import torch
import unidecode
import random


def grab_data(split_pct):
    f = './data/input.txt'
    music_data = unidecode.unidecode(open(f).read())
    n = len(music_data)
    split_idx = int(music_data*0.8)
    return music_data[:split_idx], music_data[split_idx:]


def random_data_sample(data, batch_size):
    n = len(data)
    random_idx = random.randint(0, n - batch_size - 1)
    x = data[random_idx:random_idx + batch_size]
    y = data[random_idx + 1:random_idx + batch_size + 1]
    return x, y


def checkpoint(state, file='./saves/checkpoint.pth.tar'):
    torch.save(state, file)


def resume(model, optimizer, filepath):
    f = torch.load(filepath)
    epoch = f['epoch']
    model.load_state_dict(f['state_dict'])
    optimizer.load_state_dict(f['optimizer'])
    return model, optimizer, epoch
