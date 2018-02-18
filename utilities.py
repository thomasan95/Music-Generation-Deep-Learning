import numpy as np
import torch
import sklearn


def grab_data(batch_size=32):
    file = './data/input.txt'
    data = np.loadtxt(file)
    print(data.shape)

grab_data()