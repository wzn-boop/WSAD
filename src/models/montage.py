import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.core.base_model import BaseDeepAD
from src.models.mountings import mounting_handler
from src.models.negative_creator import create_batch_neg


class Montage(BaseDeepAD):
    def __init__(
            self,
            # training parameters
            epochs=50, batch_size=64, lr=8e-5, seq_len=30, stride=1, epoch_steps=40,

            # network architecture parameters
            network='CDTTransformerEn', objective='OC',
            rep_dim=128, hidden_dims='100,50', act='GELU', bias=False,
            # # parameters for transformers
            n_heads=8, d_model=64, attn='cc_attn', pos_encoding='fixed', norm='LayerNorm',

            # objective parameters
            rep=True, nac=True, unc=True,

            # other parameters
            prt_steps=1, device='cuda', verbose=2, random_state=42
    ):
        super(Montage, self).__init__(
            data_type='ts', model_name='Montage', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.objective = objective

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        self.rep = rep
        self.nac = nac
        self.unc = unc

        self.c = None
        return

    def training_prepare(self, X, y):
        # check
        if 'OC' in self.objective:
            assert self.rep is True
            assert 'En' in self.network
        if 'MSE' in self.objective:
            assert self.rep is False
            assert 'AE' in self.network

        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # for i in train_loader:
        #     print(i)
        net = NetworkModule(
            network_name=self.network,
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            n_heads=self.n_heads,
            d_model=self.d_model,
            attn=self.attn,
            pos_encoding=self.pos_encoding,
            norm=self.norm,
            seq_len=self.seq_len,
            bias=False,
            rep=self.rep,  # representation head
            nac=self.nac,  # native anomaly calibration
            unc=self.unc  # uncertainty modeling-based calibration
        ).to(self.device)

        criterion_class = mounting_handler.get_objectives(self.objective)
        if 'OC' in self.objective:
            self.c = self._set_c(net, train_loader)
            criterion = criterion_class(self.c, unc=self.unc, nac=self.nac)
        else:
            criterion = criterion_class()

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)

        if self.rep is True:
            # use native anomaly calibration
            if self.nac is True:
                batch_x1, batch_y1 = self._create_negative(batch_x)
                batch_y0 = -1 * torch.ones(batch_x.shape[0]).float().to(self.device)

                y_all = torch.hstack([batch_y0, batch_y1])

                # use uncertainty calibration
                if self.unc is True:
                    rep, rep_duplicate, pred_batch_x = net(batch_x)
                    _, _, pred_batch_x1 = net(batch_x1)
                    pred_all = torch.cat([pred_batch_x, pred_batch_x1]).view(-1)
                    loss = criterion(rep=rep, rep2=rep_duplicate, y=y_all, pred=pred_all)

                else:
                    rep, pred_batch_x = net(batch_x)
                    _, pred_batch_x1 = net(batch_x1)
                    pred_all = torch.cat([pred_batch_x, pred_batch_x1]).view(-1)
                    loss = criterion(rep=rep, y=y_all, pred=pred_all)

            else:
                if self.unc is True:
                    rep, rep_duplicate = net(batch_x)
                    loss = criterion(rep=rep, rep2=rep_duplicate)
                else:
                    rep = net(batch_x)
                    loss = criterion(rep=rep)

        # reconstruction-based methods
        else:
            batch_x_new, hidden = net(batch_x)
            loss = criterion(batch_x, batch_x_new)

        return loss

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)

        if self.rep is True:
            if self.unc is True:
                if self.nac is True:
                    rep, rep_duplicate, pred_batch_x = net(batch_x)
                else:
                    rep, rep_duplicate = net(batch_x)

                dis1 = torch.sum((rep - self.c) ** 2, dim=1)
                dis2 = torch.sum((rep_duplicate - self.c) ** 2, dim=1)
                score = dis1 + dis2

            else:
                if self.nac is True:
                    rep, pred = net(batch_x)
                else:
                    rep = net(batch_x)
                score = torch.sum((rep - self.c) ** 2, dim=1)

        # reconstruction-based methods
        else:
            batch_x_new, rep = net(batch_x)
            score = criterion(batch_x[:, -1], batch_x_new[:, -1])
            score = torch.mean(score, dim=1)
            rep = rep[:, -1, :]

        return rep, score

    def _set_c(self, net, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        net.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                out = net(x)
                if type(out) is tuple:
                    z = out[0]
                else:
                    z = out
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # if c is too close to zero, set to +- eps
        # a zero unit can be trivially matched with zero weights
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def _create_negative(self, batch_x):
        neg_batch_size = int(0.2 * self.batch_size)
        neg_cand_idx = np.random.randint(0, batch_x.shape[0], neg_batch_size)
        batch_x1, batch_y1 = create_batch_neg(batch_seqs=batch_x[neg_cand_idx],
                                              max_cut_ratio=0.5,
                                              seed=np.random.randint(1e+5),
                                              return_mul_label=False,
                                              ss_type='FULL')
        batch_x1, batch_y1 = batch_x1.to(self.device), batch_y1.to(self.device)
        return batch_x1, batch_y1


class NetworkModule(torch.nn.Module):
    def __init__(
            self, network_name, n_features, hidden_dims='100,50', rep_dim=64,
            n_heads=8, d_model=64, attn='self_attn', pos_encoding='fixed', norm='BatchNorm', seq_len=100,
            activation='ReLU', bias=False,
            rep=True, nac=True, unc=True
    ):
        super(NetworkModule, self).__init__()

        self.rep = rep
        self.nac = nac
        self.unc = unc

        if rep:
            assert 'AE' not in network_name, 'use non-AE networks for representation learning model '
        else:
            assert 'AE' in network_name, 'use AE networks for reconstruction model'

        network_params = {
            'n_features': n_features,
            'n_hidden': hidden_dims,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        if 'Transformer' in network_name:
            network_params['n_heads'] = n_heads
            network_params['d_model'] = d_model
            network_params['attn'] = attn
            network_params['pos_encoding'] = pos_encoding
            network_params['norm'] = norm
            network_params['seq_len'] = seq_len
        elif 'ConvSeq' in network_name:
            network_params['seq_len'] = seq_len

        temporal_network_class = mounting_handler.get_network(network_name)
        self.temporal_network = temporal_network_class(**network_params)

        self.act = torch.nn.LeakyReLU()

        if rep:
            self.rep_l1 = torch.nn.Linear(rep_dim, rep_dim, bias=bias)
            self.rep_l2 = torch.nn.Linear(rep_dim, rep_dim, bias=bias)

        if self.unc:
            self.rep_l1_dup = torch.nn.Linear(rep_dim, rep_dim, bias=bias)

        if self.nac:
            self.clf_l1 = torch.nn.Linear(rep_dim, rep_dim, bias=bias)
            self.clf_l2 = torch.nn.Linear(rep_dim, 1, bias=bias)

        return

    def forward(self, x):
        x = self.temporal_network(x)

        if self.rep:
            rep = self.rep_l2(self.act(self.rep_l1(x)))

            if self.nac:
                score = self.clf_l2(self.act(self.clf_l1(x)))
                if self.unc:
                    rep_duplicate = self.rep_l2(self.act(self.rep_l1_dup(x)))
                    return rep, rep_duplicate, score
                else:
                    return rep, score
            else:
                if self.unc:
                    rep_duplicate = self.rep_l2(self.act(self.rep_l1_dup(x)))
                    return rep, rep_duplicate
                else:
                    return rep
        else:
            return x


if __name__ == '__main__':
    x = np.random.randn(2000, 20)
    model = Montage(

        # ------------------------------ One-class --------------------- #

        # network='CDTTransformerEn',
        # rep_dim=128, hidden_dims='512', act='GELU', bias=False,
        # n_heads=8, d_model=64, attn='cc_attn', pos_encoding='fixed', norm='LayerNorm',

        # network='TransformerEn',
        # rep_dim=128, hidden_dims='512', act='GELU', bias=False,
        # n_heads=8, d_model=64, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',

        # network='TCNEn',
        # rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,

        # network='ConvSeqEn',
        # rep_dim=128, hidden_dims='100', act='ReLU', bias=False,

        # network='LSTMEn',
        # rep_dim=128, hidden_dims='100', act='ReLU', bias=False,

        # network='GRUEn',
        # rep_dim=128, hidden_dims='100', act='ReLU', bias=False,

        #####
        # objective='OC',
        # epochs=10, batch_size=32, lr=8e-5,
        # rep=True, nac=True, unc=True,

        # ------------------------------ reconstruction --------------------- #

        # network='TCNAE',
        # rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
        network='LSTMEn',
        rep_dim=128, hidden_dims='100', act='ReLU', bias=False,

        # objective = 'MSE',
        # epochs = 2, batch_size = 32, lr = 8e-5,
        # rep = False

        objective='OC',
        epochs=10, batch_size=32, lr=8e-5,
        rep=True, nac=True, unc=True

    )
    model.fit(x)
    s = model.decision_function(x)
    print(s.shape)
    # # #
    # model = Montage(network='CDTTransformerEn', objective='OC', rep=True, nac=True, unc=True)
    # model.fit(x)
