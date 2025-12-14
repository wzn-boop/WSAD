import numpy as np


def linear_smooth(x, L):
    return int(x * L)


def linear_reward(x, L):
    return -1.0 * x / L + 1.0


def smooth_PA(gt_label, pre_label, reward):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    **
    :param reward: TP weight
    :param pre_label - predicted label
    :param gt_label - ground-truth label
    """
    pre_label = pre_label.copy()
    reward = reward.copy()

    assert len(pre_label) == len(gt_label)
    splits = np.where(gt_label[1:] != gt_label[:-1])[0] + 1
    splits = np.append(splits, len(gt_label))

    is_anomaly = gt_label[0] == 1
    pos = 0

    for sp in splits:
        if is_anomaly and 1 in pre_label[pos:sp]:
            first_index = np.where(pre_label[pos:sp] == 1)[0][0] + pos
            # print(first_index, sp)
            num_ano = np.where(pre_label[pos:sp] == 1)[0].size
            len_seg = sp - pos
            x = float(num_ano / len_seg)

            num_PA = linear_smooth(x, L=(sp - first_index))
            cur = first_index
            count = 0
            while cur < sp and count < num_PA:
                if pre_label[cur] == 0:
                    pre_label[cur] = 1
                    reward[cur] = linear_reward(cur - pos, len_seg)
                    count += 1
                else:
                    reward[cur] = linear_reward(first_index - pos, len_seg)
                cur += 1

        is_anomaly = not is_anomaly
        pos = sp

    return pre_label, reward