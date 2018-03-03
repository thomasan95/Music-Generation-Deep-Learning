import torch.nn as nn
from torch.autograd import Variable
import torch


class GRU(nn.Module):

    def __init__(self, batch_size, hidden_units=100, num_layers=1, num_outputs=26, dropout=0):
        """
        :param batch_size: size of input expecting
        :param hidden_units: number of hidden units to use, default 100
        :param num_layers: number of layers, default 1
        :param num_outputs: output to make to (26 for ABC)
        """
        super(GRU, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Create GRU network
        self.gru = nn.GRU(input_size=1,
                          hidden_size=hidden_units,
                          num_layers=num_layers,
                          dropout=dropout)

        self.dense = nn.Linear(hidden_units, num_outputs)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initializes the hidden state weight matrix of the GRU network
        :return: Initialized weight matrix
        """
        if torch.cuda.is_available():
            return nn.init.xavier_normal(Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_units),
                                                  requires_grad=True).cuda())
        else:
            return nn.init.xavier_normal(Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_units),
                                                  requires_grad=True))

    def forward(self, x):
        """
        :param x: input of size (seq_len, batch, input_size)
        :param h: hidden state of previous cell or initial cell
        :param c: initial cell state for each layer in batch
        :return char_out: character outputs from network
        """
        # Use 1's because we want character level
        # x = self.emb(x.view(1, -1)) # 1 x Batch Size
        out, self.hidden = self.gru(x.view(1, -1, 1), self.hidden)
        char_out = self.dense(out)
        return char_out
