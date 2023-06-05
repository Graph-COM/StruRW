import os.path as osp
import os
import nni
import json
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.data import Data
from graph_conv import GraphConv, MixUpGCNConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import sys
import pdb
import numpy as np
import random
import copy
import argparse
import matplotlib.pyplot as plt

sys.path.append('..')
from Utils.pre_data import datasets
from Utils import utils, edge_rw

parser = argparse.ArgumentParser('Mixup')
parser.add_argument('-d', '--dataset', type=str, help='dataset used', required=True)
parser.add_argument('-m', '--method', type=str, help='method used', required=True)
parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)
parser.add_argument('--mixup', type=bool, help='Whether to have Mixup', default=False)
parser.add_argument("--epochs", type=int, help='number of epochs', default=300)
parser.add_argument('--dir_name', type=str, required=True)
parser.add_argument("--seed", type=int, help='number of random seed', default=5)
parser.add_argument("--hidden_channels", type=int, default=50)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.007)
parser.add_argument('--best_model', type=bool,help='if use best model from validation results', default=True)
parser.add_argument('--valid_data', type=str, help='src means use source valid to select model, tgt means use target valid', default='tgt')

parser.add_argument('--reweight', type=bool, help='if we want to reweight the source graph', default=False)
parser.add_argument('--rw_freq', type=int, help='every number of epochs to compute the new weights based on psuedo-labels', default=15)
parser.add_argument('--start_epoch', type=int, help='starting epoch for reweighting', default=200)
parser.add_argument('--pseudo', type=bool, help='if use pseudo label for reweighting', default=False)
parser.add_argument("--rw_lmda", type=float, default=1)
parser.add_argument("--use_valid_label", type=bool, help='if we want to use target validation ground truth label in rw', default=True)

parser.add_argument('--src_name', type=str, default='dblp')
parser.add_argument('--tgt_name', type=str, default='acm')
parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2005)
parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2007)
parser.add_argument('--domain_split', type=str, default='word')
parser.add_argument("--ps", type=float, help="the source intraclass connection probability for CSBM dataset", default=0.2)
parser.add_argument("--qs", type=float, help="the source interclass connection probability for CSBM dataset", default=0.02)
parser.add_argument("--pt", type=float, help="the target intraclass connection probability for CSBM dataset", default=0.2)
parser.add_argument("--qt", type=float, help="the target interclass connection probability for CSBM dataset", default=0.16)


torch.set_num_threads(5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimized_params = nni.get_next_parameter()
"""
optimized_params = {
        'epochs': 300,
        'lr': 0.007,
        'opt_decay_step': 50,
        'opt_decay_rate': 0.8,
        'hidden_dim': 128,
        'start_epoch': 200,
        'rw_freq': 15,
        'rw_lmda': 1,
    }
"""
print(optimized_params)
args = parser.parse_args()
if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

args.reweight = False
if 'rw' in args.method:
    args.reweight = True
    args.rw_freq = optimized_params['rw_freq']
    args.start_epoch = optimized_params['start_epoch']

args.epochs = optimized_params['epochs']
args.lr = optimized_params['lr']
args.hidden_channels = optimized_params['hidden_dim']
args.rw_lmda = optimized_params['rw_lmda']
args.dropout = optimized_params['dropout']

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    #data.y[data.val_id] = -1
    #data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype=torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype=torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data


def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    #data.train_id = data.train_id.cpu()
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel, args):
        super(Net, self).__init__()
        self.conv1 = MixUpGCNConv(in_channel, hidden_channels)
        self.conv2 = MixUpGCNConv(hidden_channels, hidden_channels)
        self.conv3 = MixUpGCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)
        self.rw_lmda = args.rw_lmda
        self.dropout = args.dropout

    def forward(self, x0, edge_index, edge_index_b, lam, id_new_value_old, edge_weight):
        x1 = self.conv1(x0, x0, edge_index, edge_weight, self.rw_lmda)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, x1, edge_index, edge_weight, self.rw_lmda)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x0_b = x0[id_new_value_old]
        x1_b = x1[id_new_value_old]
        x2_b = x2[id_new_value_old]

        x0_mix = x0 * lam + x0_b * (1 - lam)

        new_x1 = self.conv1(x0, x0_mix, edge_index, edge_weight, self.rw_lmda)
        new_x1_b = self.conv1(x0_b, x0_mix, edge_index_b, edge_weight, self.rw_lmda)
        new_x1 = F.relu(new_x1)
        new_x1_b = F.relu(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=self.dropout, training=self.training)

        new_x2 = self.conv2(x1, x1_mix, edge_index, edge_weight, self.rw_lmda)
        new_x2_b = self.conv2(x1_b, x1_mix, edge_index_b, edge_weight, self.rw_lmda)
        new_x2 = F.relu(new_x2)
        new_x2_b = F.relu(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=self.dropout, training=self.training)

        #new_x3 = self.conv3(x2, edge_index, x2_mix)
        #new_x3_b = self.conv3(x2_b, edge_index_b, x2_mix)
        #new_x3 = F.relu(new_x3)
        #new_x3_b = F.relu(new_x3_b)

        #x3_mix = new_x2 * lam + new_x2_b * (1 - lam)
        #x3_mix = F.dropout(x3_mix, p=0.4, training=self.training)

        x = x2_mix
        x = self.lin(x)
        return x.log_softmax(dim=-1)

# func train one epoch
def train(data, model):
    model.train()

    if args.mixup:
        lam = np.random.beta(4.0, 4.0)
    else:
        lam = 1.0

    data_b, id_new_value_old = shuffleData(data)
    data = data.to(device)
    data_b = data_b.to(device)

    optimizer.zero_grad()
    #print(torch.unique(data_b.y))
    out = model(data.x, data.edge_index, data_b.edge_index, lam, id_new_value_old, data.edge_weight)
    loss = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam + \
           F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam)

    loss.backward()
    optimizer.step()

    return loss.item()


# test
@torch.no_grad()
def test(data, model, domain):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device), data.edge_index.to(device), 1, np.arange(data.num_nodes), data.edge_weight.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    if domain == "src":
        for _, id_ in data('train_id', 'val_id', 'test_id'):
            accs.append(correct[id_].sum().item() / id_.shape[0])
    else:
        for _, id_ in data('tgt_val_id', 'tgt_test_id'):
            accs.append(correct[id_].sum().item() / id_.shape[0])
    return accs

def get_avg_std_report(reports):
    all_keys = {k: [] for k in reports[0]}
    avg_report, avg_std_report = {}, {}
    for report in reports:
        for k in report:
            if report[k]:
                all_keys[k].append(report[k])
            else:
                all_keys[k].append(0)
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.5f} +/- {np.std(v):.5f}' for k, v in all_keys.items()}
    return avg_report, avg_std_report

# set random seed
print(args)
num_seeds = args.seed
#all_seeds = np.random.choice(range(5), num_seeds, replace=False)
all_seeds = [1, 3, 5, 6, 8]
reports = []
for seed in all_seeds:
    SEED = seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    if args.dataset == "SBM":
        # stochastic block model
        data_src = datasets.scbm_multi(args.num_nodes, args.sigma, args.ps, args.qs)
        data_tgt = datasets.scbm_multi(args.num_nodes, args.sigma, args.pt, args.qt)
    elif args.dataset == "dblp_acm":
        data_src = datasets.prepare_dblp_acm("../../StruRW_dataset", args.src_name)
        data_tgt = datasets.prepare_dblp_acm("../../StruRW_dataset", args.tgt_name)
    elif args.dataset == "cora":
        dataset = datasets.prepare_cora("../../StruRW_dataset", args.domain_split, "covariate")
        data_src = dataset
        data_tgt = dataset
    elif args.dataset == "arxiv":
        if args.domain_split == "degree":
            dataset = datasets.prepare_arxiv("../../StruRW_dataset", args.domain_split)
            data_src = dataset
            data_tgt = dataset
        else:
            data_src = datasets.prepare_arxiv("../../StruRW_dataset", [args.start_year, args.end_year])
            data_tgt = datasets.prepare_arxiv("../../StruRW_dataset", [2018, 2020])
    else:
        print("Other datasets")

    data_src.train_id = data_src.source_training_mask
    data_src.val_id = data_src.source_validation_mask
    data_src.test_id = data_src.source_testing_mask
    data_tgt.tgt_val_id = data_tgt.target_validation_mask
    data_tgt.tgt_test_id = data_tgt.target_testing_mask
    data_src = data_src.to(device)
    data_tgt = data_tgt.to(device)

    # define model
    model = Net(hidden_channels=args.hidden_channels, in_channel=data_src.num_node_features, out_channel=data_src.num_classes, args = args).to(device)
    best_model = Net(hidden_channels=args.hidden_channels, in_channel=data_src.num_node_features, out_channel=data_src.num_classes, args = args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    directory = args.dir_name
    parent_dir = "./"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)
    if isdir == False:
        os.mkdir(path)

    sys.stdout = utils.Logger(path)

    best_valid_score = 0
    best_epoch = 0
    accord_epoch = 0
    accord_train_acc = 0
    accord_train_loss = 0
    src_edge_weight = data_src.edge_weight.to(device)
    str_diff = []
    rec_rw = 0
    for epoch in range(0, args.epochs):
        str_diff_i = 0
        if args.reweight and epoch >= (args.start_epoch - 1):
            if args.pseudo:
                pred_tgt = model(data_tgt.x.to(device), data_tgt.edge_index.to(device), data_tgt.edge_index.to(device), 1, np.arange(data_tgt.num_nodes), data_tgt.edge_weight.to(device))
                pred_tgt = pred_tgt.argmax(dim=1)
                pred_tgt = pred_tgt.to(device)
                data_tgt.y_hat = pred_tgt
                if args.use_valid_label:
                    data_tgt.y_hat[data_tgt.target_validation_mask] = data_tgt.y[data_tgt.target_validation_mask]

                if (epoch - args.start_epoch - 1) % args.rw_freq == 0:
                    edge_rw.calculate_reweight(data_src, data_tgt, args.dataset, args.domain_split)
                _, tgt_diff = edge_rw.calculate_str_diff(data_src, data_tgt, args.dataset, args.domain_split)

            else:
                tgt_diff = 0
                if epoch == args.start_epoch - 1:
                    edge_rw.calculate_reweight(data_src, data_tgt, args.dataset, args.domain_split)
                    src_tgt_diff, _ = edge_rw.calculate_str_diff(data_src, data_tgt, args.dataset, args.domain_split)
                    print(src_tgt_diff)

            if tgt_diff:
                str_diff_i = tgt_diff.item()

        rw_done = 1
        all_1 = torch.ones(data_src.num_edges).to(device)
        if torch.equal(data_src.edge_weight, all_1):
            rw_done = 0

        if epoch >= args.start_epoch:
            str_diff.append(str_diff_i)

        rec_rw += rw_done

        loss = train(data_src, model)
        accs = test(data_src, model, "src")
        accs_tgt = test(data_tgt, model, "tgt")
        inter_report = {'acc_tgt_valid': accs_tgt[0],
                  'acc_tgt_test': accs_tgt[1],
                  'acc_src_valid': accs[1],
                  'acc_src_train': accs[0],
                  'acc_src_test': accs[2],
                  'default': accs_tgt[1]}

        if args.valid_data == "src":
            valid_score = inter_report['acc_src_valid']
        else:
            valid_score = inter_report['acc_tgt_valid']
        if valid_score > best_valid_score:
            best_epoch = epoch
            best_valid_score = valid_score
            best_model = model
            #torch.save(model.state_dict(), path + '/best_valid_model_' + str(seed) + '.pt')

        #nni.report_intermediate_result(inter_report)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, src valid Acc: {accs[1]:.4f}, '
              f'tgt valid acc: {accs_tgt[0]:.4f}, tgt test Acc: {accs_tgt[1]:.4f}')

    print("rw times: " + str(rec_rw))
    print("best_epoch: " + str(best_epoch))
    if args.best_model == False:
        best_model = model
    #else:
     #   best_model = Net(hidden_channels=args.hidden_channels, in_channel=data_src.num_node_features, out_channel=data_src.num_classes, args = args)
      #  best_model.load_state_dict(torch.load(path + '/best_valid_model_' + str(seed) + '.pt'))
       # best_model = best_model.to(device)

    plt.plot(np.arange(args.start_epoch, args.epochs), str_diff, label='str_diff')
    plt.xlabel('Epochs')
    plt.ylabel('str_diff with truth')
    plt.legend()
    filename = path + "/str_diff" + ".png"
    plt.savefig(filename)
    plt.clf()

    accs = test(data_src, best_model, "src")
    accs_tgt = test(data_tgt, best_model, "tgt")
    report = {'acc_tgt_valid': accs_tgt[0],
              'acc_tgt_test': accs_tgt[1],
              'acc_src_valid': accs[1],
              'acc_src_train': accs[0],
              'acc_src_test': accs[2],
              'default': accs_tgt[1]}
    reports.append(report)
    print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report, indent=4)), print('-' * 80), print('-' * 80)

avg_report, avg_std_report = get_avg_std_report(reports)
nni.report_final_result(avg_report)
print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)


