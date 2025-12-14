from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch
from deepod.utils.utility import get_sub_seqs_label
from torch.utils.data import DataLoader, TensorDataset

import utils_general
from src.models.core.base_model import BaseDeepAD
from src.models.mountings import mounting_handler
from src.models.negative_creator import create_batch_neg
from utils_general import evaluate_window_size

torch.autograd.set_detect_anomaly(True)


class Ensemble(BaseDeepAD):
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
        super(Ensemble, self).__init__(
            data_type='ts', model_name='Ensemble', epochs=epochs, batch_size=batch_size, lr=lr,
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

    # def multi_dataset(self, X)：
    def training_prepare(self, X, y):
        tensor_list = [torch.tensor(arr) for arr in X]
        dataset = TensorDataset(*tensor_list)

        # dataset = TensorDataset(torch.from_numpy(X[0]).float(), torch.from_numpy(X[1]).float())
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=my_collate_fn)
        # for batch in train_loader:
        #     print(batch)
        net = NetworkModule(
            network_name=self.network,
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            # n_heads=self.n_heads,
            # d_model=self.d_model,
            # attn=self.attn,
            # pos_encoding=self.pos_encoding,
            # norm=self.norm,
            # seq_len=self.seq_len,
            bias=False,
        ).to(self.device)

        criterion_class = mounting_handler.get_objectives(self.objective)
        criterion = criterion_class('MSE')
        # criterion = F.mse_loss()
        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion):
        # batch_x = batch_x.float().to(self.device)
        # batch_x_clone = [i.clone() for i in batch_x]
        batch_x = [tensor.float().to(self.device) for tensor in batch_x]
        # batch_x = batch_x.to(self.device)
        # reconstruction-based methods
        batch_x_new = net(batch_x)
        loss = 0
        for i in range(len(batch_x)):
            loss += criterion(batch_x[i], batch_x_new[i])
        # return torch.mean(torch.stack(loss), dim=0)
        return loss

    def inference_prepare(self, X):
        tensor_list = [torch.tensor(arr) for arr in X]
        dataset = TensorDataset(*tensor_list)

        # dataset = TensorDataset(torch.from_numpy(X[0]).float(), torch.from_numpy(X[1]).float())
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=my_collate_fn)
        self.criterion.reduction = 'none'
        return test_loader

    def inference_forward(self, batch_x, net, criterion):

        batch_x = [tensor.float().to(self.device) for tensor in batch_x]
        score = []
        batch_x_new = net(batch_x)
        for j in range(len(batch_x)):
            score.append(criterion(batch_x[j][:, -1], batch_x_new[j][:, -1]))

        rep = batch_x_new
        score = torch.sum(torch.cat(score, dim=1), dim=1)
        return rep, score

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

    def decision_function(self, X, return_rep=False):
        testing_n_samples = X.shape[0]

        X_seqs, n_features = get_sub_seqs_ensemble(X, stride=1)

        representations = []
        s_final = np.zeros(testing_n_samples)

        self.test_loader = self.inference_prepare(X_seqs)

        z, scores = self._inference()
        z, scores = self.decision_function_update(z, scores)

        if self.data_type == 'ts':
            padding = np.zeros(testing_n_samples-len(scores))
            scores = np.hstack((padding, scores))

        s_final += scores
        representations.extend(z)
        # representations = np.array(representations)

        if return_rep:
            return s_final, representations
        else:
            return s_final

    def fit(self, X, y=None):
        X_seqs, self.n_features = get_sub_seqs_ensemble(X, stride=self.stride)
        y_seqs = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.stride) if y is not None else None
        self.train_data = X_seqs
        self.train_label = y_seqs
        self.n_samples = X_seqs[0].shape[0]

        if self.verbose >= 1:
            print('Start Training...')

        if self.n_ensemble == 'auto':
            self.n_ensemble = int(np.floor(100 / (np.log(self.n_samples) + self.n_features)) + 1)
        if self.verbose >= 1:
            print(f'ensemble size: {self.n_ensemble}')

        for _ in range(self.n_ensemble):
            self.train_loader, self.net, self.criterion = self.training_prepare(self.train_data,
                                                                                y=self.train_label)
            self._training()

        if self.verbose >= 1:
            print('Start Inference on the training data...')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return self


class NetworkModule(torch.nn.Module):
    def __init__(
            self, network_name, n_features, hidden_dims='100,50', rep_dim=64,
            # n_heads=8, d_model=64, attn='self_attn', pos_encoding='fixed', norm='BatchNorm', seq_len=100,
            activation='ReLU', bias=False,
    ):
        super(NetworkModule, self).__init__()

        self.net_lst = []
        self.feature_vector_lst = []
        # self.ensemble_foction = LSTMEncoder(n_features, n_hidden=hidden_dims, n_output=rep_dim)
        self.device = torch.device('cuda')
        self.decoder = None  # 在forward函数中初始化

        encoder_params = {
            'n_features': 1,
            'n_hidden': hidden_dims,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        decoder_params = {
            'n_features': int(hidden_dims),
            'n_hidden': '1',
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }
        temporal_network_class = mounting_handler.get_network(network_name)
        self.encoder = temporal_network_class(**encoder_params)
        self.decoder = temporal_network_class(**decoder_params)
        self.act = torch.nn.LeakyReLU()
        return

    def forward(self, x):
        feature_vector_lst = []
        for i in range(len(x)):
            # xi = self.net_lst[i](x[i])
            xi, _ = self.encoder(x[i])
            # decoder = LSTMEncoder(x[i].shape[2], n_hidden='100', n_output=x[i].shape[1], ).to(device=self.device)
            # repeatxi = xi.repeat(x[i].size(1), 1, 1).permute(1, 0, 2)
            decoded, _ = self.decoder(xi)
            feature_vector_lst.append(decoded)
        # combined_features = torch.cat(self.feature_vector_lst, dim=2)  # 不同维度向量形状不同不能stack
        # combined_features = combined_features.view(, 1, -1)  # 调整形状以匹配 LSTM 的输入要求
        # prediction = self.ensemble_foction(combined_features)
        return feature_vector_lst


def get_sub_seqs(x_arr, seq_len=100, stride=1, n=0):
    x_arr = x_arr.values

    # 在x_arr前添加n个0
    x_arr_with_zeros = np.pad(x_arr, (n, 0), 'constant')

    seq_starts = np.arange(0, x_arr_with_zeros.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr_with_zeros[i:i + seq_len] for i in seq_starts])

    return x_seqs


def my_collate_fn(batch):
    merged_tensors = [torch.stack(items, dim=0) for items in zip(*batch)]
    for i in range(len(merged_tensors)):
        merged_tensors[i] = torch.unsqueeze(merged_tensors[i], 2)
    return merged_tensors


def get_sub_seqs_ensemble(X, stride=1):
    # X (n_samples, n_futures)
    # window_sizes = []
    column_names = X.columns
    column_names_list = column_names.tolist()
    # tmp = X[column_names_list[0]]
    # for i in range(X.shape[1]):
    #     window_size_candidates = evaluate_window_size(X[column_names_list[i]], 'ACF')
    #     print(window_size_candidates)
    #     window_size = max(set(window_size_candidates), key=window_size_candidates.count)
    #     window_sizes.append(window_size)
    #     print(window_size)
    window_sizes = evaluate_window_size(X, 'ACF')
    print(window_sizes)
    min_window_size = min(window_sizes)  # 用来计算补零的数量
    X_seqs = []
    X_seqs_shapes = []

    for j in range(X.shape[1]):
        num_zeros = window_sizes[j] - min_window_size
        X_seqs_j = get_sub_seqs(X[column_names_list[j]], seq_len=window_sizes[j], stride=stride, n=num_zeros)
        X_seqs.append(X_seqs_j)
        X_seqs_shapes.append(X_seqs_j.shape)
    # X_seqs = np.stack(X_seqs, axis=2)

    # 创建一个defaultdict来存储相同形状的元素的下标集合
    shape_indices = defaultdict(list)

    # 遍历shapes列表，并将具有相同形状的元素的下标存储在shape_indices中
    for idx, shape in enumerate(X_seqs_shapes):
        shape_indices[shape].append(idx)
    X_seqs_new = []
    for shape, indices in shape_indices.items():
        if len(indices) > 1:
            X_seqs_new.append(np.stack([X_seqs[i] for i in indices], axis=2))
        else:
            X_seqs_new.append(X_seqs[indices[0]])

    # print(shape_indices)
    # 将合并后的数据存储到TensorDataset中
    # dataset_list = [TensorDataset(torch.from_numpy(tmp)) for tmp in X_seqs_new]
    # 打印存储的TensorDataset
    # for dataset in dataset_list:
    #     print(dataset)
    features = X.shape[1]
    return X_seqs, features


if __name__ == '__main__':
    data = 'ASD'
    # data_root = '/home/wzn/_processed_data/'
    data_root = r'E:/sys/Desktop/data/_processed_data/'
    entities = 'FULL'
    train_df_lst, test_df_lst, label_lst, name_lst = utils_general.get_data_lst(data, data_root, entities)
    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
        # x_sqes = get_sub_seqs_ensemble(train)
        model = Ensemble(
            network='LSTMEn',
            rep_dim=64, hidden_dims='64', act='ReLU', bias=False,

            objective='MSE',
            epochs=10, batch_size=64, lr=8e-5,
            rep=True, nac=True, unc=True
        )

        model.fit(train)
        scores = model.decision_function(test)

