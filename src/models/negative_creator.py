import torch
import numpy as np
import random


def create_batch_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL'):
    """
    create a batch of negative samples based on the input sequences,
    the output batch size is the same as the input batch size
    :param batch_seqs: input sequences
    :param max_cut_ratio:
    :param seed:
    :param return_mul_label:
    :param ss_type:
    :return:
    """
    rng = np.random.RandomState(seed=seed)

    batch_size, l, dim = batch_seqs.shape
    cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)
    n_cut_dim = rng.randint(1, dim+1, size=batch_size)
    cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

    if type(batch_seqs) == np.ndarray:
        batch_neg = batch_seqs.copy()
        neg_labels = np.zeros(batch_size, dtype=int)
    else:
        batch_neg = batch_seqs.clone()
        neg_labels = torch.LongTensor(batch_size)


    if ss_type != 'FULL':
        pool = rng.randint(1e+6, size=int(1e+4))
        if ss_type == 'collective':
            pool = [a % 6 == 0 or a % 6 == 1 for a in pool]
        elif ss_type == 'contextual':
            pool = [a % 6 == 2 or a % 6 == 3 for a in pool]
        elif ss_type == 'point':
            pool = [a % 6 == 4 or a % 6 == 5 for a in pool]
        flags = rng.choice(pool, size=batch_size, replace=False)
    else:
        flags = rng.randint(1e+5, size=batch_size)

    n_types = 6
    for ii in range(batch_size):
        flag = flags[ii]

        # collective anomalies
        if flag % n_types == 0:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
            neg_labels[ii] = 1

        elif flag % n_types == 1:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
            neg_labels[ii] = 1

        # contextual anomalies
        elif flag % n_types == 2:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
            neg_labels[ii] = 2

        elif flag % n_types == 3:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
            neg_labels[ii] = 2

        # point anomalies
        elif flag % n_types == 4:
            batch_neg[ii, -1, cut_dim[ii]] = 2
            neg_labels[ii] = 3

        elif flag % n_types == 5:
            batch_neg[ii, -1, cut_dim[ii]] = -2
            neg_labels[ii] = 3

    if return_mul_label:
        return batch_neg, neg_labels
    else:
        neg_labels = torch.ones(batch_size).float()
        return batch_neg, neg_labels


def create_batch_classes(batch_seqs, transformation='flip',
                         transform_net=None,
                         seed=0):
    """
    create a batch of synthetic samples by time-series transformation
    :param batch_seqs: input sequences, torch.Tensor
    :param transformation: transformation operation
    :param transform_net: torch.Module
    :param seed:
    :return:
    """
    batch_size, l, dim = batch_seqs.shape

    # if type(batch_seqs) == np.ndarray:
    #     new_batch = batch_seqs.copy()
    #     if transformation == 'flip':
    #         new_batch = new_batch[:, ::-1, :]
    #     elif transformation == 'gaussian':
    #         noise = np.random.normal(0, 0.1, new_batch.shape)
    #         new_batch = new_batch + noise
    # else:

    new_batch = batch_seqs.clone()
    if transformation == 'flip':
        new_batch = torch.flip(new_batch, dims=[1])
    elif transformation == 'gaussian':
        noise = torch.randn(new_batch.shape)
        new_batch = new_batch + noise

    if transformation == 'neural':
        assert transform_net is not None
        new_batch = transform_net(new_batch)

    return new_batch
