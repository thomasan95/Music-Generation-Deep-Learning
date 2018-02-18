import torch.nn as nn

class LSTM_network(nn.Module):

    def __init__(self, batch_size, hidden_units, num_layers=1):
        super().__init__()
        cell = nn.LSTM(input_size=batch_size,
                       hidden_size=hidden_units,
                       num_layers=num_layers)
    def forward(self, x, (h, c)):
        '''
        :param x: input of size (seq_len, batch, input_size)
        :param h: hidden state of previous cell or initial cell
        :param c: initial cell state for each layer in batch
        :return output:(seq_len, batch_ hidden_size*num_directions) output features from last layer for each t
        :return hn: tensor for hidden state t
        :return cn: tensor for cell state of t
        '''
        output, hn, cn = self.cell(x, (h, c))
        return output, hn, cn
