import torch
from deepod.core.network_utility import _handle_n_hidden, _instantiate_class


class GRUEncoder(torch.nn.Module):
    def __init__(self, n_features, n_hidden='20', n_output=20,
                 bias=False, dropout=None, activation='ReLU'):
        super(GRUEncoder, self).__init__()

        hidden_dim, n_layers = _handle_n_hidden(n_hidden)

        if dropout is None:
            dropout = 0.0

        self.gru = torch.nn.GRU(n_features, hidden_dim, n_layers,
                                batch_first=True,
                                bias=bias,
                                dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        out, h = self.gru(x)
        out = self.fc(out[:, -1])
        return out



class LSTMEncoder(torch.nn.Module):
    def __init__(self, n_features, n_hidden='20', n_output=20,
                 bias=False, dropout=None, activation='ReLU'):
        super(LSTMEncoder, self).__init__()

        hidden_dim, n_layers = _handle_n_hidden(n_hidden)

        if dropout is None:
            dropout = 0.0

        self.lstm = torch.nn.LSTM(n_features, hidden_size=hidden_dim,
                                  batch_first=True,
                                  bias=bias,
                                  dropout=dropout,
                                  num_layers=n_layers)
        self.fc = torch.nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        out, (hn, c) = self.lstm(x)
        # out = self.fc(out[:, -1])
        return out, hn

#
# if __name__ == '__main__':
#     x = np.random.randn(2000, 20)
