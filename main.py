import argparse
import os
import time

import numpy as np

import utils_general
from src.evaluation.base import get_metrics, get_metrics_sPA
from src.evaluation.pa import pa_adjust_scores
from src.models.montage import Montage
# from src.models.lstmensemble import LstmEnsemble
from utils_general import evaluate_window_size, evaluate_window_size_ensemble

parser = argparse.ArgumentParser()
# parser.add_argument('--data_root', type=str, default=f'/home/zxh/tsad/5-TSdata/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'D:/tsad/5-TSdata/_processed_data/')
parser.add_argument('--data_root', type=str, default=f'E:/sys/Desktop/data/_processed_data/')
# parser.add_argument('--data_root', type=str, default=f'/home/wzn/_processed_data/')
parser.add_argument('--data', type=str,
                    default='ASD',
                    help='dataset name')
parser.add_argument('--output_dir', type=str, default='./&results/')
parser.add_argument("--entities", type=str,
                    default='FULL',
                    # default='C-1',
                    # default='omi-1',
                    help='FULL represents all the entities, or a list of entity names split by comma')
parser.add_argument('--network', type=str, default='LSTMEn',
                    choices=['TCNAE', 'LATAE', 'GRUEn', 'LSTMEn', 'TCNEn', 'ConvSeqEn', 'TransformerEn',
                             'CDTTransformerEn', 'LATTransformerEn'])
parser.add_argument('--rep', help='', type=str, default='True')
parser.add_argument('--objective', help='', type=str, default='OC',
                    choices=['MSE', 'OC'])
parser.add_argument('--nac', help='', type=bool, default=True)
parser.add_argument('--unc', help='', type=bool, default=True)

parser.add_argument('--stride', help='', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=30)

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--epoch_steps', type=int, default=40)

parser.add_argument('--rep_dim', help='', type=int, default=128)
parser.add_argument('--hidden_dims', help='', type=str, default='100,50')
parser.add_argument('--act', help='', type=str, default='ReLU')
parser.add_argument('--pe', help='', type=str, default='fixed')
parser.add_argument('--attn', help='', type=str, default='cc_attn')
parser.add_argument('--lr', help='', type=float, default=0.00005)
parser.add_argument('--batch_size', help='', type=int, default=64)
parser.add_argument('--bias', help='', type=bool, default=False)

args = parser.parse_args()
model_configs = {
    'network': args.network,  # variable
    'objective': args.objective,  # variable
    'rep': False if args.rep == 'False' else True,  # search
    'nac': args.nac,
    'unc': args.unc,

    'epochs': args.num_epochs,  # search
    'epoch_steps': args.epoch_steps,  # search
    'batch_size': args.batch_size,  # search
    'lr': args.lr,  # search

    'seq_len': args.seq_len,
    'stride': args.stride,

    'rep_dim': args.rep_dim,
    'hidden_dims': args.hidden_dims,
    'act': args.act,
}

datasets = args.data.split(',')
for dataset in datasets:
    print(dataset)
    result_dir = os.path.join(args.output_dir, f'{args.network}_{dataset}/')
    os.makedirs(result_dir, exist_ok=True)
    if args.network == 'CDTTransformerEn':
        result_file = os.path.join(result_dir,
                                   f'attn_{args.attn}_epochs_{args.num_epochs}_bs_{args.batch_size}_lr_{args.lr}_sl_{args.seq_len}_hd_{args.hidden_dims}_rep_{args.rep_dim}.csv')
    else:
        result_file = os.path.join(result_dir,
                                   f'epochs_{args.num_epochs}_bs_{args.batch_size}_lr_{args.lr}_sl_{args.seq_len}_hd_{args.hidden_dims}_rep_{args.rep_dim}.csv')

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f = open(result_file, 'a') 
    print(f'Time: {cur_time}, Data: {args.data} \n Configs: {model_configs} \n ', file=f)
    print(f'network, data, auroc, aupr, f1, p, r,,'
          f'adj_auroc, adj_aupr, adj_f1, adj_p, adj_r,,'
          f's_adj_auroc, s_adj_aupr, s_adj_f1, s_adj_p, s_adj_r', file=f)
    f.close()

    train_df_lst, test_df_lst, label_lst, name_lst = utils_general.get_data_lst(args.data, args.data_root,
                                                                                args.entities)
    eval_metrics_lst = []
    adj_eval_metrics_lst = []
    sPA_eval_metrics_lst = []

    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
        print(f'\n\n Running {args.network} on {name}')
        # configs['seed'] = 46 + i
        window_size_candidates = evaluate_window_size(train, 'ACF')
        print(window_size_candidates)
        # window_size_candidates = evaluate_window_size_ensemble(train)

        # #can select different window selection methods, or write a loop to get all window sizes.
        # FFT, ACF, SuSS, MWF, Autoperiod, RobustPeriod. Human means set window size by human
        model_configs['seq_len'] = window_size_candidates[0]
        # #select based on actual situation, each window size is calculated according to each independent channel
        model = Montage(**model_configs)
        model.fit(train)

        scores = model.decision_function(test)


        eval_metrics = get_metrics(label, scores)
        adj_eval_metrics = get_metrics(label, pa_adjust_scores(label, scores))
        sPA_eval_metrics = get_metrics_sPA(label, scores)
        eval_metrics_lst.append(eval_metrics)
        adj_eval_metrics_lst.append(adj_eval_metrics)
        sPA_eval_metrics_lst.append(sPA_eval_metrics)
        # save evaluation metrics raw results
        txt = f'{args.network}, {name},'
        txt += ', '.join(['%.4f' % a for a in eval_metrics]) + ', pa, ' + ', '.join(
            ['%.4f' % a for a in adj_eval_metrics]) + ', spa, ' + ', '.join(['%.4f' % a for a in sPA_eval_metrics])

        f = open(result_file, 'a')
        print(txt)
        print(txt, file=f)
        f.close()
    avg_eval = np.average(np.array(eval_metrics_lst), axis=0)
    adj_eval = np.average(np.array(adj_eval_metrics_lst), axis=0)
    sPA_eval = np.average(np.array(sPA_eval_metrics_lst), axis=0)
    txt = f'{args.network}, avg,'
    txt += ', '.join(['%.4f' % a for a in avg_eval]) + ', pa, ' + ', '.join(
        ['%.4f' % a for a in adj_eval]) + ', spa, ' + ', '.join(['%.4f' % a for a in sPA_eval])
    f = open(result_file, 'a')
    print(txt)
    print(txt, file=f)
    f.close()
