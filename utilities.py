import numpy as np
import torch
import unidecode
import random

def grab_data():
    f = './data/input.txt'
    music_data = unidecode.unidecode(open(f).read())
    return music_data

def random_data_sample(data, batch_size):
    n = len(data)
    random_idx = random.randint(0, n - batch_size - 1)
    x = data[random_idx:random_idx + batch_size]
    y = data[random_idx + 1:random_idx + batch_size + 1]
    return x, y
