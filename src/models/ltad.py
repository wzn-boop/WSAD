import math
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import RandomState
from torch.utils.data import DataLoader
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.core.networks.ts_network_tcn import TcnResidualBlock
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_transformer import TSTransformerEncoder


class LTAD(BaseDeepAD):
    """
            LTAD class for Calibrated One-class classifier for Unsupervised Time series Anomaly detection

            Parameters
            ----------
            sequence_length: integer, default=100
                sliding window length
            stride: integer, default=1
                sliding window stride
            num_epochs: integer, default=40
                the number of training epochs
            batch_size: integer, default=64
                the size of mini-batches
            lr: float, default=1e-4
                learning rate
            hidden_dims: integer or list of integer, default=16,
                the number of neural units in the hidden layer
            emb_dim: integer, default=16
                the dimensionality of the feature space
            rep_hidden: integer, default=16
                the number of neural units of the hidden layer
            kernel_size: integer, default=2
                the size of the convolutional kernel in TCN
            dropout: float, default=0
                the dropout rate
            bias: bool, default=True
                the bias term of the linear layer
            es: bool, default=False
                early stopping
            seed: integer, default=42
                random state seed
            device: string, default='cuda'
            logger: logger or print, default=None
            model_dir: string, default='couta_model/'
                directory to store intermediate model files
            """
    def __init__(self, seq_len=100, stride=1, n_pairs=5, skip=1, loss_type='auto', label_type='auto', prior=False,
                 epochs=40, batch_size=64, lr=1e-4,
                 hidden_dims=16, rep_dims=16, n_output=20,
                 kernel_size=2, dropout=0.0, bias=True, train_val_pc=0.25,
                 epoch_steps=-1, prt_steps=5, device='cuda',
                 verbose=2, random_state=42
                 ):
        super(LTAD, self).__init__(
            model_name='LTAD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.rep_dims = rep_dims
        self.loss_type = loss_type
        if self.loss_type == 'MSE':
            self.label_type = label_type
        else:
            self.label_type = 'auto'

        if self.loss_type == 'MSE':
            self.emb_dims = 1
        else:
            self.emb_dims = n_pairs
        self.bias = bias
        self.prior = prior

        self.n_output = n_output
        self.train_val_pc = train_val_pc
        self.n_pairs = n_pairs
        self.skip = skip

        self.net = None
        self.criterion = None
        self.test_df = None
        self.test_labels = None

        return

    def fit(self, X, y=None):
        """
        Fit detector.

        Parameters
        ----------
        X: dataframe of pandas
            input training set
        """
        self.n_features = X.shape[1]
        sequences = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        if self.prior:
            sequences_pairs = self.mate_seq_pairs_prior(sequences, self.skip)
            print('prior')
        else:
            sequences_pairs = self.mate_seq_pairs(sequences, self.skip)
            print('forward')
        print(X.shape, sequences.shape, sequences_pairs.shape)

        if self.train_val_pc > 0:
            train_seqs = sequences_pairs[: -int(self.train_val_pc * len(sequences))]
            val_seqs = sequences_pairs[-int(self.train_val_pc * len(sequences)):]
        else:
            train_seqs = sequences_pairs
            val_seqs = None

        self.net = TsNetModule(
            input_dim=self.n_features,
            stride=self.stride,
            n_pairs=self.n_pairs,
            seq_len=self.seq_len,
            label_type=self.label_type,
            skip=self.skip,
            hidden_dims=self.hidden_dims,
            emb_dim=self.emb_dims,
            rep_hidden=self.rep_dims,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            linear_bias=self.bias,
            tcn_bias=self.bias
        )

        self.net.to(self.device)

        self.net = self.train(self.net, train_seqs, val_seqs)

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return

    def decision_function(self, X, return_rep=False):
        """
        Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

        Parameters
        ----------
            X: pd.DataFrame
                testing dataframe

        Returns
        -------
            predictions_dic: dictionary of predicted results
            The anomaly score of the input samples.
        """
        test_sub_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        seqs_len = test_sub_seqs.shape[0]
        if self.prior:
            test_sub_seqs = self.mate_seq_pairs_prior(test_sub_seqs, self.skip)
        else:
            test_sub_seqs = self.mate_seq_pairs(test_sub_seqs, self.skip)
        seqs_pairs_len = test_sub_seqs.shape[0]

        test_dataset = SubseqData(test_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        self.net.eval()
        with torch.no_grad():
            score_lst = []
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                s = self.criterion(x_output[0], x_output[1], reduction='none')
                score_lst.append(s)

        scores = torch.cat(score_lst).data.cpu().numpy()
        if self.prior:
            scores_pad = np.hstack([0 * np.ones(seqs_len - seqs_pairs_len), scores])
            scores_pad = np.hstack([0 * np.ones(self.seq_len - 1), scores_pad])
        else:
            scores_pad = np.hstack([0 * np.ones(self.seq_len - 1), scores])
            scores_pad = np.hstack([scores_pad, 0 * np.ones(seqs_len - seqs_pairs_len)])

        return scores_pad

    def train(self, net, train_seqs, val_seqs=None):
        val_loader = DataLoader(dataset=SubseqData(val_seqs),
                                batch_size=self.batch_size,
                                drop_last=False, shuffle=False) if val_seqs is not None else None
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        self.criterion = TsLoss(loss_type=self.loss_type, reduction='mean')

        net.train()
        for i in range(self.epochs):
            train_loader = DataLoader(dataset=SubseqData(train_seqs),
                                      batch_size=self.batch_size,
                                      drop_last=True, pin_memory=True, shuffle=True)

            loss_lst = []
            for ii, x0 in enumerate(train_loader):
                x0 = x0.float().to(self.device)
                x0_output = net(x0)
                rep, y = x0_output[0], x0_output[1]
                loss = self.criterion(rep, y)  # mean

                net.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)

            epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()

            # validation phase
            val_loss = np.NAN
            if val_seqs is not None:
                val_loss = []
                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        x_out = net(x)
                        loss = self.criterion(x_out[0], x_out[1])
                        val_loss.append(loss)
                val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            if (i+1) % self.prt_steps == 0:
                print(
                    f'|>>> epoch: {i+1:02}  |   loss: {epoch_loss:.6f}, '
                    f'val_loss: {val_loss:.6f}'
                )

        return net

    def mate_seq_pairs(self, seq, skip):
        # seq's shape [seq_num, seq_len, dim]
        x_pairs = []
        seq_num = seq.shape[0]
        # print(seq_num, self.n_pairs)
        for i in range(seq_num):
            if i + skip * self.n_pairs <= seq_num - 1:
                pairs = [seq[i]]
                for j in range(self.n_pairs):
                    pairs.append(seq[i + skip * (j + 1)])
                x_pairs.append(pairs)
        x_pairs = np.array(x_pairs)
        return x_pairs

    def mate_seq_pairs_prior(self, seq, skip):
        # seq's shape [seq_num, seq_len, dim]
        x_pairs = []
        seq_num = seq.shape[0]
        # print(seq_num, self.n_pairs)
        for i in range(seq_num):
            if i + skip * self.n_pairs <= seq_num - 1:
                pairs = [seq[i + skip * self.n_pairs]]
                for j in range(self.n_pairs):
                    pairs.append(seq[i + skip * j])
                x_pairs.append(pairs)
        x_pairs = np.array(x_pairs)
        return x_pairs

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


class SubseqData(Dataset):
    def __init__(self, x, y=None, w1=None, w2=None):
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight1[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]


class TsNetModule(torch.nn.Module):
    def __init__(self, input_dim, stride, n_pairs, seq_len, label_type, skip,
                 hidden_dims=32, rep_hidden=32, emb_dim=1,
                 kernel_size=2, dropout=0.2,
                 tcn_bias=True, linear_bias=True):
        super(TsNetModule, self).__init__()

        self.layers = []
        self.stride = stride
        self.n_pairs = n_pairs
        self.seq_len = seq_len
        self.label_type = label_type
        self.skip = skip

        if type(hidden_dims) == int: hidden_dims = [hidden_dims]
        elif type(hidden_dims) == str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TcnResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             padding=padding_size, dropout=dropout, bias=tcn_bias)]
        self.network = torch.nn.Sequential(*self.layers)

        self.l1 = torch.nn.Linear(2 * hidden_dims[-1], rep_hidden, bias=linear_bias)
        self.l2 = torch.nn.Linear(rep_hidden, emb_dim, bias=linear_bias)
        self.act = torch.nn.LeakyReLU()


    def forward(self, x):
        x = x.reshape(-1, x.shape[2], x.shape[3])
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        out_pairs, y = self.con_pairs(out, self.stride, self.seq_len)
        rep = self.l2(self.act(self.l1(out_pairs)))  # rep's shape [batch_size * num, emb_dim]

        return rep, y

    def con_pairs(self, out, stride, seq_len):
        # input out's size is (batch_size * (num + 1), hidden_dims[-1])
        dim = out.shape[1]
        out = out.reshape(-1, self.n_pairs + 1, dim)
        batch_size = out.shape[0]
        out_pairs = torch.empty(batch_size, self.n_pairs, 2 * dim).to(out)
        for i in range(self.n_pairs):
            out_pairs[:, i] = torch.cat((out[:, 0],  out[:, (i + 1)]), dim=1)
        out_pairs = out_pairs.reshape(-1, 2 * dim)
        y = torch.arange(self.n_pairs).unsqueeze(0).expand((batch_size, self.n_pairs))  # shape [n_trans] -> [1, n_trans] -> [n_samples, n_trans]

        if self.label_type == 'auto':
            y = y
        elif self.label_type == 'stride':
            y = (y + 1) * stride * self.skip
        elif self.label_type == 'overlap':
            y = (seq_len - (y + 1) * stride * self.skip)

        y = y.to(out)  # y shape [batch_size, n_pairs]

        return out_pairs, y


class TranNetModule(torch.nn.Module):
    def __init__(self, input_dim, stride, n_pairs, seq_len, label_type,
                 hidden_dims=32, n_output=20,
                 rep_hidden=32, emb_dim=1,
                 linear_bias=True):
        super(TranNetModule, self).__init__()

        self.layers = []
        self.stride = stride
        self.n_pairs = n_pairs
        self.seq_len = seq_len
        self.label_type = label_type

        self.network = TSTransformerEncoder(n_features=input_dim, n_output=n_output,
                                            seq_len=self.seq_len,
                                            d_model=hidden_dims, n_hidden='512')

        self.l1 = torch.nn.Linear(2 * n_output, rep_hidden, bias=linear_bias)
        self.l2 = torch.nn.Linear(rep_hidden, emb_dim, bias=linear_bias)
        self.act = torch.nn.LeakyReLU()


    def forward(self, x):
        x = x.reshape(-1, x.shape[2], x.shape[3])
        out = self.network(x)
        out_pairs, y = self.con_pairs(out, self.stride, self.seq_len)
        # rep's size is (batch_size*num, emd_dim)
        rep = self.l2(self.act(self.l1(out_pairs)))

        return rep, y

    def con_pairs(self, out, stride, seq_len):
        # input out's size is (batch_size * (num + 1), hidden_dims[-1])
        dim = out.shape[1]
        out = out.reshape(-1, self.n_pairs + 1, dim)
        batch_size = out.shape[0]
        out_pairs = torch.empty(batch_size, self.n_pairs, 2 * dim).to(out)
        for i in range(self.n_pairs):
            out_pairs[:, i] = torch.cat((out[:, 0], out[:, i + 1]), dim=1)
        out_pairs = out_pairs.reshape(-1, 2 * dim)
        y = torch.arange(self.n_pairs).unsqueeze(0).expand(
            (batch_size, self.n_pairs))  # shape [n_trans] -> [1, n_trans] -> [n_samples, n_trans]

        if self.label_type == 'auto':
            y = y
        elif self.label_type == 'stride':
            y = (y + 1) * stride
        elif self.label_type == 'overlap':
            y = (seq_len - (y + 1) * stride) / seq_len

        y = y.to(out)  # y shape [batch_size, n_pairs]

        return out_pairs, y


class TsLoss(torch.nn.Module):
    def __init__(self, loss_type, reduction='mean'):
        super(TsLoss, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'MSE':
            self.CrossELoss = torch.nn.MSELoss(reduction='none')
        else:
            self.CrossELoss = torch.nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction

    def forward(self, rep, y, reduction=None):
        batch_size = y.shape[0]  # y shape [batch_size, n_pairs], rep shape [batch_size * n_pairs, emb_dim]
        loss = None

        # MSE
        if self.loss_type == 'MSE':
            loss = self.CrossELoss(rep.reshape(batch_size, -1, rep.shape[-1]), y.reshape(batch_size, -1, 1).float())#.unsqueeze(dim=1) (batch_size, num, 1)
            loss = loss.mean(1)
        else:  # CrossEntropy
            loss = self.CrossELoss(rep, y.reshape(-1, 1).squeeze(1).long())  #(batch_size * num, )， y can be one-hot or index
            loss = loss.reshape(batch_size, -1, 1).mean(1)  # (batch_size, num, 1) ->  (batch_size, 1)

        loss = loss.mean(1)   #(batch_size, )

        if reduction is None:
            reduction = self.reduction
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction == 'none':
            loss = loss

        return loss


if __name__=='__main__':
    from testbed.utils import import_ts_data_unsupervised
    from deepod.metrics import ts_metrics, point_adjustment

    data_dir = '/home/xuhz/dataset/5-TSdata/_processed_data/'
    data = 'UCR_natural_mars'

    entities = 'FULL'
    model_name = 'LTAD'
    train_df_lst, test_df_lst, label_lst, name_lst = import_ts_data_unsupervised(data_dir, data, entities=entities)
    print(name_lst)
    num_runs = 5

    f1_lst = []
    aupr_lst = []
    train_num = 0
    test_num = 0
    anomaly_num = 0
    N = len(name_lst)
    f_len = 0
    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
        entries = []
        train_num = train_num + train.shape[0]
        test_num = test_num + test.shape[0]
        anomaly_num = anomaly_num + np.sum(label)
        f_len = train.shape[1]
    ratio = anomaly_num / (train_num + test_num)
    print(train_num / N, test_num / N, f_len, anomaly_num / N, ratio, N)
    print(f'{train_num / N:.0f}, {test_num / N:.0f}, {f_len:.0f}, {anomaly_num / N:.0f}, {ratio:.3f}')