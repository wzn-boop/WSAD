# -*- coding: utf-8 -*-
"""
testbed of unsupervised time series anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import getpass
import warnings
import yaml
import time
import importlib as imp
import numpy as np
import utils_general
# from utils import import_ts_data_unsupervised
from deepod.metrics import ts_metrics, point_adjustment

from math import log, floor

import pandas as pd
import tqdm
from scipy.optimize import minimize

from src.models.lstmensemble import Ensemble
from src.models.ltad import LTAD
from src.flad import FLAD
"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

	    Parameters
	    ----------
	    q
		    Detection level (risk)

	    Returns
	    ----------
    	SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm

        data : numpy.array
		    data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
	    ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        verbose: bool
            (default True) If True, prints log
        min_extrema bool
            (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            # self.init_data = -self.init_data
            # self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm
        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        if len(self.peaks) < 5:
            S_unique = np.flip(np.unique(S))
            i = 1
            while len(np.unique(self.peaks)) < 5:
                if i == len(S_unique):
                    break
                self.init_threshold = S_unique[i]
                self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
                i = i + 1

        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
		    scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
        Run SPOT on the stream

        Parameters
        ----------
        with_alarm : bool
		    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'

            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms

        """
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

def pot_eval(init_score, score, q=1e-4, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    if len(np.unique(init_score)) == 1:
        return init_score[0]
    else:
        s = SPOT(q)  # SPOT object
        s.fit(init_score, score)  # data import
        # s.initialize(level=level, min_extrema=True)  # initialization step
        s.initialize(level=level, min_extrema=True)
        # ret = s.run(dynamic=False)  # run
        ret = s.run(dynamic=True)
        print(len(ret['alarms']))
        print(len(ret['thresholds']))
        # pot_th = -np.mean(ret['thresholds'])
        pot_th = np.mean(ret['thresholds'])
        return pot_th

def get_anomaly_label(train_anomaly_scores, anomaly_scores, name=None):
    # if name == 'USAD' or name == 'COUTA' or name == 'RAN':
    #     th = 0.5
    # else:
    th = pot_eval(train_anomaly_scores, anomaly_scores)
    anomaly_label = np.where(anomaly_scores > th, 1, 0)
    # anomaly_index = np.where(anomaly_label)[0]
    # for i in anomaly_index:
    #     anomaly_label[i: i + 20] = 1
    return anomaly_label


parser = argparse.ArgumentParser()  # 创建一个解析器对象
# parser.add_argument('--data_root', type=str, default=f'/home/zxh/tsad/5-TSdata/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'D:/tsad/5-TSdata/_processed_data/')
parser.add_argument('--data_root', type=str, default=f'E:/sys/Desktop/data/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'/home/wzn/_processed_data/')  # 设置数据根目录，默认值为特定路径
parser.add_argument('--data', type=str,  # 设置数据集名称，默认值为特定数据集名称
                    default='ASD',
                    help='dataset name')
parser.add_argument('--output_dir', type=str, default='./&results/')  # 设置结果输出目录，默认值为特定路径
parser.add_argument("--entities", type=str,  # 设置实体，默认值为完整集合
                    default='FULL',
                    # default='C-1',
                    # default='omi-1',
                    help='FULL represents all the entities, or a list of entity names split by comma')
parser.add_argument('--network', type=str, default='LSTMEn',  # 设置网络模型，默认值为特定模型
                    choices=['TCNAE', 'LATAE', 'GRUEn', 'LSTMEn', 'TCNEn', 'ConvSeqEn', 'TransformerEn',
                             'CDTTransformerEn', 'LATTransformerEn'])
parser.add_argument('--rep', help='', type=str, default='True')  # 设置是否使用表示学习，默认为True
parser.add_argument('--objective', help='', type=str, default='MSE',  # 设置目标函数，默认值为特定函数
                    choices=['MSE', 'OC'])
parser.add_argument('--nac', help='', type=bool, default=True)  # 设置是否使用非线性激活，默认为True
parser.add_argument('--unc', help='', type=bool, default=True)  # 设置是否使用不确定性估计，默认为True

parser.add_argument('--stride', help='', type=int, default=1)  # 设置步长，默认为1
parser.add_argument('--seq_len', type=int, default=30)  # 设置序列长度，默认为30

parser.add_argument('--num_epochs', type=int, default=10)  # 设置训练迭代次数，默认为10
parser.add_argument('--epoch_steps', type=int, default=40)  # 设置每个迭代次数的步数，默认为40

parser.add_argument('--rep_dim', help='', type=int, default=64)  # 设置表示学习的维度，默认为128
parser.add_argument('--hidden_dims', help='', type=str, default='100')  # 设置隐藏层维度，默认为特定值
parser.add_argument('--act', help='', type=str, default='ReLU')  # 设置激活函数，默认为ReLU
parser.add_argument('--pe', help='', type=str, default='fixed')  # 设置位置编码方式，默认为固定编码
parser.add_argument('--attn', help='', type=str, default='cc_attn')  # 设置注意力机制，默认为特定值
parser.add_argument('--lr', help='', type=float, default=0.00005)  # 设置学习率，默认为特定值
parser.add_argument('--batch_size', help='', type=int, default=64)  # 设置批处理大小，默认为64
parser.add_argument('--bias', help='', type=bool, default=False)  # 设置偏置，默认为False
parser.add_argument('--model', help='', type=str, default='FLAD')
parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--dataset", type=str,
                    # default='DASADS',
                    # default='MSLs',
                    # default='SMAP',
                    default='ASD',
                    # default='SWaT_cut',
                    help='dataset name or a list of names split by comma')
parser.add_argument("--runs", type=int, default=1,
                    help="how many times we repeat the experiments to "
                         "obtain the average performance")
parser.add_argument("--note", type=str, default='')

args = parser.parse_args()  # 解析命令行参数

# module = imp.import_module('deepod.models')
# model_class = getattr(module, args.model)

# path = 'configs.yaml'
# with open(path) as f:
#     d = yaml.safe_load(f)
#     try:
#         model_configs = d[args.model]
#     except KeyError:
#         print(f'config file does not contain default parameter settings of {args.model}')
#         model_configs = {}
# model_configs = {
#     'network': args.network,  # 变量
#     'objective': args.objective,  # 变量
#     'rep': False if args.rep == 'False' else True,  # 搜索
#     'nac': args.nac,
#     'unc': args.unc,
#
#     'epochs': args.num_epochs,  # 搜索
#     'epoch_steps': args.epoch_steps,  # 搜索
#     'batch_size': args.batch_size,  # 搜索
#     'lr': args.lr,  # 搜索
#
#     'seq_len': args.seq_len,
#     'stride': args.stride,
#
#     'rep_dim': args.rep_dim,
#     'hidden_dims': args.hidden_dims,
#     'act': args.act,
# }
model_configs = {}
# model_configs['seq_len'] = args.seq_len
# model_configs['stride'] = args.stride

print(f'Model Configs: {model_configs}')

# # setting result file/folder path
cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
result_file = os.path.join(args.output_dir, f'{args.model}.csv')

# # print header in the result file
if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, dataset: {args.dataset}, '
          f'{args.runs}runs, {cur_time}', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
    print(f'Note: {args.note}', file=f)
    print(f'---------------------------------------------------------', file=f)
    print(f'data, average_lead, time, model', file=f)
    f.close()

dataset_name_lst = args.dataset.split(',')

for dataset in dataset_name_lst:
    # # read data
    if dataset == 'MSL':
        anormly_ratio = 1
    elif dataset == 'SMAP':
        anormly_ratio = 0.85
    else:
        anormly_ratio = 0.9
    # data_pkg = import_ts_data_unsupervised(dataset_root,
    #                                        dataset, entities=args.entities,
    #                                        combine=args.entity_combined)
    data_pkg =utils_general.get_data_lst(dataset, args.data_root,args.entities)
    train_lst, test_lst, label_lst, name_lst = data_pkg

    entity_metric_lst = []
    entity_metric_std_lst = []
    entity_t_lst = []
    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
        entries = []
        t_lst = []
        runs = args.runs

        for j in range(runs):
            start_time = time.time()
            print(f'\nRunning [{j + 1}/{args.runs}] of [{args.model}] on Dataset [{dataset}-{dataset_name}]')

            t1 = time.time()

            # clf = model_class(**model_configs, random_state=42 + j)
            clf = FLAD(**model_configs)
            clf.fit(train_data)
            train_scores = clf.decision_function(train_data)
            test_scores = clf.decision_function(test_data)

            combined_energy = np.concatenate([train_scores, test_scores], axis=0)
            thresh = np.percentile(combined_energy, 100 - anormly_ratio)
            anomaly_label = (test_scores > thresh).astype(int)

            # anomaly_label = get_anomaly_label(train_scores, test_scores)

            anom_index = np.where(anomaly_label == 1)[0]
            tmp_seg = []
            anom_pairs = []
            true_pairs = []
            for i in anom_index:
                tmp_seg.append(i)
                if i + 1 not in anom_index:
                    anom_pairs.append((tmp_seg[0], tmp_seg[-1]))
                    tmp_seg = []
            # anom_time = [(str(anomaly_label.index[pair[0]]), str(anomaly_label.index[pair[1]])) for pair in anom_pairs][1:]

            true_index = np.where(labels == 1)[0]
            true_seg = []
            for i in true_index:
                true_seg.append(i)
                if i + 1 not in true_index:
                    true_pairs.append((true_seg[0], true_seg[-1]))
                    true_seg = []
            average_lead = 0
            # last_interval = 0
            last_value = 0
            event_len = len(true_pairs)
            for anom in anom_pairs:
                drop = False
                for index, true in enumerate(true_pairs):
                    if index ==0:
                        last_interval = true_pairs[index][0] - last_value
                    else:
                        last_interval = true_pairs[index][0] - true_pairs[index-1][1]
                    # if anom[0] in range(true[0] - min(last_interval, model_configs['seq_len']//2), true[1]):
                    if anom[0] in range(true[0] - last_interval, true[1]):
                        average_lead += (true[0] - anom[0])
                        drop = True
                    # elif (true[0] - min(last_interval, model_configs['seq_len']//2)) in range(anom[0], anom[1]):
                    elif (true[0] - last_interval) in range(anom[0], anom[1]):
                        # average_lead += min(last_interval, model_configs['seq_len']//2)
                        average_lead += last_interval
                        drop = True

                    if drop:
                        # true_pairs.pop([i for i in range(0,index)])
                        last_value = true_pairs[index][1]
                        del true_pairs[:index+1]
                        break
            for true in true_pairs:
                average_lead += (true[0] - true[1])
            average_lead = average_lead/event_len

            t = time.time() - t1


            # print single results
            txt = f'{dataset}-{dataset_name},'
            txt +=f', {average_lead}'
            txt += f', model, {args.model}, time, {t:.1f} s, runs, {j + 1}/{args.runs}'
            print(txt)

            entries.append(average_lead)
            t_lst.append(t)

        avg_entry = np.average(entries)
        std_entry = np.std(entries)
        t_entry = np.average(t_lst)

        entity_metric_lst.append(avg_entry)
        entity_metric_std_lst.append(std_entry)
        entity_t_lst.append(t_entry)

        f = open(result_file, 'a')
        txt = '%s, %.4f, %.1f, %s, %s ' % \
              (dataset_name, avg_entry, t_entry, args.model, str(model_configs))
        print(txt)
        print(txt, file=f)
        f.close()

    avg_entities = np.average(entity_metric_lst)
    std_entities = np.average(entity_metric_std_lst)
    t_entities = np.average(entity_t_lst)
    f = open(result_file, 'a')
    txt = '%s, %.4f, %.1f, %s, %s ' % \
          (dataset, avg_entities, t_entities, args.model, str(model_configs))
    print(txt)
    print(txt, file=f)
    f.close()
