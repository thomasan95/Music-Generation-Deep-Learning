import torch.nn as nn
from torch.autograd import Variable
import torch


class LSTM(nn.Module):

    def __init__(self, batch_size, hidden_units=100, num_layers=1, num_outputs=26):
        '''
        :param batch_size: size of input expecting
        :param hidden_units: number of hidden units to use, default 100
        :param num_layers: number of layers, default 1
        :param num_outputs: output to make to (26 for ABC)
        '''
        super(LSTM, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Create LSTM network
        self.emb = nn.Embedding(batch_size, hidden_units)
        self.lstm = nn.LSTM(input_size=batch_size,
                            hidden_size=hidden_units,
                            num_layers=num_layers)
        self.dense = nn.Linear(hidden_units, num_outputs)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (nn.init.xavier_normal(Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))),
                nn.init.xavier_normal(Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))))

    def forward(self, x, h, c):
        '''
        :param x: input of size (seq_len, batch, input_size)
        :param h: hidden state of previous cell or initial cell
        :param c: initial cell state for each layer in batch
        :return char_out: character outputs from network
        '''
        # Use 1's because we want character level
        x = self.emb(x.view(1, -1))
        out, self.hidden = self.lstm(x.view(1, 1, -1), self.hidden)
        char_out = self.dense(out)
        return char_out
