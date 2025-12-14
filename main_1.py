import argparse  # 用于解析命令行参数
import os  # 用于处理文件和目录相关的操作
import time  # 用于处理时间相关的操作

import numpy as np  # 用于处理数组和矩阵计算

import utils_general  # 用于处理一些通用的函数和工具

from src.evaluation.base import get_metrics, get_metrics_sPA  # 用于计算评估指标
from src.evaluation.pa import pa_adjust_scores  # 用于调整评分
from src.models.montage import Montage  # 自定义的模型Montage
from src.models.mountings.networks_class import LSTMEncoder
from src.models.lstmensemble import Ensemble
from utils_general import evaluate_window_size, evaluate_window_size_ensemble  # 用于评估窗口大小

parser = argparse.ArgumentParser()  # 创建一个解析器对象
# parser.add_argument('--data_root', type=str, default=f'/home/zxh/tsad/5-TSdata/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'D:/tsad/5-TSdata/_processed_data/')
parser.add_argument('--data_root', type=str, default=f'E:/sys/Desktop/data/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'/home/wzn/_processed_data/')  # 设置数据根目录，默认值为特定路径
parser.add_argument('--data', type=str,  # 设置数据集名称，默认值为特定数据集名称
                    default='MSL',
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

args = parser.parse_args()  # 解析命令行参数

model_configs = {
    'network': args.network,  # 变量
    'objective': args.objective,  # 变量
    'rep': False if args.rep == 'False' else True,  # 搜索
    'nac': args.nac,
    'unc': args.unc,

    'epochs': args.num_epochs,  # 搜索
    'epoch_steps': args.epoch_steps,  # 搜索
    'batch_size': args.batch_size,  # 搜索
    'lr': args.lr,  # 搜索

    'seq_len': args.seq_len,
    'stride': args.stride,

    'rep_dim': args.rep_dim,
    'hidden_dims': args.hidden_dims,
    'act': args.act,
}

datasets = args.data.split(',')  # 将数据集名称按逗号分隔为列表

for dataset in datasets:  # 遍历数据集列表
    print(dataset)  # 打印数据集名称
    result_dir = os.path.join(args.output_dir, f'{args.network}_{dataset}/')  # 构建结果目录路径
    os.makedirs(result_dir, exist_ok=True)  # 创建结果目录（如果不存在）

    if args.network == 'CDTTransformerEn':  # 判断网络模型是否为CDTTransformerEn
        result_file = os.path.join(result_dir,
                                   f'attn_{args.attn}_epochs_{args.num_epochs}_bs_{args.batch_size}_lr_{args.lr}_sl_{args.seq_len}_hd_{args.hidden_dims}_rep_{args.rep_dim}.csv')  # 构建结果文件路径
    else:
        result_file = os.path.join(result_dir,
                                   f'epochs_{args.num_epochs}_bs_{args.batch_size}_lr_{args.lr}_sl_{args.seq_len}_hd_{args.hidden_dims}_rep_{args.rep_dim}.csv')  # 构建结果文件路径

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 获取当前时间
    f = open(result_file, 'a')  # 打开结果文件，以追加模式写入
    print(f'Time: {cur_time}, Data: {args.data} \n Configs: {model_configs} \n ', file=f)  # 在结果文件中写入时间、数据和配置信息
    print(f'network, data, auroc, aupr, f1, p, r,,'
          f'adj_auroc, adj_aupr, adj_f1, adj_p, adj_r,,'
          f's_adj_auroc, s_adj_aupr, s_adj_f1, s_adj_p, s_adj_r', file=f)  # 在结果文件中写入指标名称
    f.close()  # 关闭结果文件

    train_df_lst, test_df_lst, label_lst, name_lst = utils_general.get_data_lst(args.data, args.data_root,  # 获取训练集、测试集、标签和名称列表
                                                                                args.entities)  # 按照指定实体获取数据

    eval_metrics_lst = []  # 创建空列表用于存储评估指标
    adj_eval_metrics_lst = []  # 创建空列表用于存储调整后的评估指标
    sPA_eval_metrics_lst = []  # 创建空列表用于存储sPA评估指标

    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):  # 遍历数据集列表

        print(f'\n\n Running {args.network} on {name}')  # 打印正在运行的网络模型和数据集名称
        # #select based on actual situation, each window size is calculated according to each independent channel
        # model = LSTMEncoder(**model_configs)  # 创建LSTM模型
        model = Ensemble(**model_configs)  # 创建Montage模型
        model.fit(train)  # 模型训练
        scores = model.decision_function(test)  # 模型预测

        eval_metrics = get_metrics(label, scores)  # 计算评估指标
        adj_eval_metrics = get_metrics(label, pa_adjust_scores(label, scores))  # 调整评估指标
        sPA_eval_metrics = get_metrics_sPA(label, scores)  # 计算sPA评估指标

        eval_metrics_lst.append(eval_metrics)  # 将评估指标列表添加到评估指标列表中
        adj_eval_metrics_lst.append(adj_eval_metrics)  # 将调整后的评估指标列表添加到调整后的评估指标列表中
        sPA_eval_metrics_lst.append(sPA_eval_metrics)  # 将sPA评估指标列表添加到sPA评估指标列表中

        # save evaluation metrics raw results
        txt = f'{args.network}, {name},'  # 构建评估指标结果字符串
        txt += ', '.join(['%.4f' % a for a in eval_metrics]) + ', pa, ' + ', '.join(
            ['%.4f' % a for a in adj_eval_metrics]) + ', spa, ' + ', '.join(['%.4f' % a for a in sPA_eval_metrics])
        f = open(result_file, 'a')  # 打开结果文件，以追加模式写入
        print(txt, file=f)  # 在结果文件中写入平均评估指标结果
        f.close()  # 关闭结果文件

    avg_eval = np.average(np.array(eval_metrics_lst), axis=0)  # 计算评估指标的平均值
    adj_eval = np.average(np.array(adj_eval_metrics_lst), axis=0)  # 计算调整后的评估指标的平均值
    sPA_eval = np.average(np.array(sPA_eval_metrics_lst), axis=0)  # 计算sPA评估指标的平均值
    txt = f'{args.network}, avg,'
    txt += ', '.join(['%.4f' % a for a in avg_eval]) + ', pa, ' + ', '.join(
        ['%.4f' % a for a in adj_eval]) + ', spa, ' + ', '.join(['%.4f' % a for a in sPA_eval])
    f = open(result_file, 'a')  # 打开结果文件，以追加模式写入
    print(txt, file=f)  # 在结果文件中写入平均评估指标结果
    f.close()  # 关闭结果文件
