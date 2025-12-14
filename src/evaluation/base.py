from sklearn import metrics
import numpy as np
import pandas as pd
from src.evaluation import spa


def get_best_f1(label, score):
    precision, recall, ths = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    best_th = ths[np.argmax(f1)]
    return best_f1, best_p, best_r, best_th


def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r, _ = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def get_metrics_sPA(gt_label, pre_score):
    DD = np.linspace(np.mean(pre_score), max(pre_score), 20, endpoint=False)

    metric = pd.DataFrame(columns=['D', 'p_sPA', 'r_sPA', 'f1_sPA', 'FPR_sPA'])
    num = 0

    for D in DD:  # TAD threshold - Delta
        pre_labels = np.zeros(gt_label.size)
        pre_labels[pre_score >= D] = 1
        Reward = np.ones(gt_label.size)

        pre_labels_sPA, Reward_sPA = spa.smooth_PA(gt_label, pre_labels, Reward)

        metric.loc[num] = [D]  + [a for a in cal_metrics(gt_label, pre_labels_sPA, Reward_sPA)]

        num += 1


    sPA_metric = metric

    auroc = cal_auc(sPA_metric['FPR_sPA'].values.tolist(), sPA_metric['r_sPA'].values.tolist())
    aupr = cal_auc(sPA_metric['r_sPA'].values.tolist(), sPA_metric['p_sPA'].values.tolist())
    best_f1 = sPA_metric['f1_sPA'].values[np.argmax(sPA_metric['f1_sPA'].values)]
    best_p = sPA_metric['p_sPA'].values[np.argmax(sPA_metric['f1_sPA'].values)]
    best_r = sPA_metric['r_sPA'].values[np.argmax(sPA_metric['f1_sPA'].values)]
    return auroc, aupr, best_f1, best_p, best_r


def cal_metrics(gt_label, pre_label, reward):
    assert len(gt_label) == len(reward)
    TP = FP = FN = TN = 0
    for i in range(len(gt_label)):
        if gt_label[i] == 1 and pre_label[i] == 1:
            TP = TP + reward[i]
        if gt_label[i] == 1 and pre_label[i] == 0:
            FN = FN + reward[i]
        if gt_label[i] == 0 and pre_label[i] == 1:
            FP = FP + reward[i]
        if gt_label[i] == 0 and pre_label[i] == 0:
            TN = TN + reward[i]

    P = TP / (TP + FP + 1e-5)
    R = TP / (TP + FN + 1e-5)
    F1 = 2 * P * R / (P + R + 1e-5)
    FPR = FP / (FP + TN + 1e-5)
    return P, R, F1, FPR


def cal_auc(x, y):
    assert len(x) == len(y)

    if 0 in x:
        z = list(set(list(zip(x, y))))
    else:
        z = list(set(list(zip(x + [0], y + [0]))))
    x_1 = [x[0] for x in z]
    y_1 = [x[1] for x in z]

    x_sort = np.sort(x_1)
    x_arg = np.argsort(x_1)

    y_sort = [y_1[i] for i in x_arg]
    # plt.plot(x, y, 'bo')
    # plt.show()

    auc = np.trapz(y_sort, x_sort)
    return auc
