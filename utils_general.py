import os
from glob import glob

import daproli as dp
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from window_size.autoperiod import autoperiod
from window_size.mwf import mwf
from window_size.period import dominant_fourier_freq, highest_autocorrelation
from window_size.robustperiod import robust_period
from window_size.suss import suss


# 假设我们有一个弱学习器类，用于评估单个时间序列的最佳窗口大小
# 这个类需要实现fit方法来确定最佳窗口大小
class WeakLearner(BaseEstimator):
    def __init__(self):
        # 初始化弱学习器
        self.best_window_size_ = None

    def fit(self, X, y=None):
        # 实现用于确定最佳窗口大小的逻辑
        # 这里的逻辑需要你根据实际情况来定义
        # 例如，你可以使用周期性分析、自相关函数等方法
        # 假设我们随机选择一个窗口大小作为示例
        # self.best_window_size_ = np.random.choice(range(1, 10))
        self.best_window_size_ = mwf(X)
        print("best window size: {}".format(self.best_window_size_))
        if self.best_window_size_ == -1:
            self.best_window_size_ = 0
        return self

    def get_best_window_size(self):
        return self.best_window_size_


def get_data_lst(data, data_root, entities=None):
    entities_lst = entities.split(',')

    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []
    if data == 'SMD' or data == 'ASD' or data == 'MSL' or data == 'SMAP' or data == 'satellite' or data == 'MBA_selected' or data == 'MSLs':
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob(os.path.join(data_root, data, m, '*test*.csv'))
            assert len(train_path) == 1 and len(test_path) == 1
            train_path, test_path = train_path[0], test_path[0]
            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
            train_df, test_df = data_standardize(train_df, test_df)  # 归一化

            train_df_lst.append(train_df)
            test_df_lst.append(test_df)
            label_lst.append(labels)
            name_lst.append(m)

    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
        train_df, test_df = data_standardize(train_df, test_df)  # 归一化

        train_df_lst.append(train_df)
        test_df_lst.append(test_df)
        label_lst.append(labels)
        name_lst.append(data)

    return train_df_lst, test_df_lst, label_lst, name_lst


def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            # @TODO: the max and min value after min-max normalization is 1 and 0, so the clip doesn't work?
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                        X_train[col].unique(),
                        X_test[col].unique()))
    return X_train, X_test


def evaluate_window_size(data, method):
    algorithms = {
        "Human": "human",
        "FFT": dominant_fourier_freq,
        "ACF": highest_autocorrelation,
        "SuSS": suss,
        "MWF": mwf,
        "Autoperiod": autoperiod,
        "RobustPeriod": robust_period,
    }
    algorithm = algorithms[method]
    print(f"Evaluating window size candidate: {method}")
    ts = data.values.T.tolist()
    ts = [np.array(t) for t in ts]
    window_sizes = dp.map(algorithm, ts, expand_args=False, ret_type=np.array, n_jobs=-1, verbose=0)

    return window_sizes

# 实现集成学习器
def evaluate_window_size_ensemble(data):
    print(f"Evaluating window size candidate: ensemble")
    ts = data.values.T.tolist()
    ts = [np.array(t) for t in ts]
    window_sizes = []

    # 创建弱学习器的集合
    weak_learners = [WeakLearner() for _ in ts]

    # 训练每个弱学习器
    for i, learner in enumerate(weak_learners):
        learner.fit(ts[i])
        window_sizes.append(learner.get_best_window_size())

    # 使用投票策略来集成弱学习器的结果，这里使用简单的众数作为示例
    # 在实际情况中，可能需要更复杂的集成策略
    window_size_counts = np.bincount(window_sizes)
    most_common_window_size = np.argmax(window_size_counts)

    # 返回最常见的窗口大小作为集成学习器的结果
    return most_common_window_size
