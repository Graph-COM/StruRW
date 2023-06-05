import torch.optim as optim
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix, fbeta_score, \
    precision_score, recall_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np
import math
import copy
import torch.nn.functional as F
import sys


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay', default=50)
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio', default=0.8)
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.', default=0)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def CE_loss(pred, label):
    label = label.type(torch.int64)
    loss = nn.CrossEntropyLoss()
    return loss(pred, label)

def BCE_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1).type(torch.float32)
    loss = nn.BCELoss()
    return loss(pred, label)

def get_acc_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return acc_score


def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        #print(torch.unique(label).size(0))
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        print("he")
        return None
        pass


def plot_scatter(X_source, y_source, X_target, y_target, name, dir):
    if len(X_source.shape) == 1:
        X_source = torch.cat((X_source.view(-1, 1), torch.zeros(X_source.shape[0], -1)))
        X_target = torch.cat((X_target.view(-1, 1), torch.zeros(X_target.shape[0], -1)))

    num_class = torch.unique(y_source)
    for i in num_class:
        i = i.item()
        plt.scatter(X_target[y_target == i][:, 0], X_target[y_target == i][:, 1],
                    label='Class_' + str(i) + '_tgt')
        plt.scatter(X_source[y_source == i][:, 0], X_source[y_source == i][:, 1],
                    label='Class_' + str(i) + '_src')
    plt.legend()
    plt.title(name)
    filename = dir + "/" + name + ".png"
    plt.savefig(filename)
    plt.clf()


def plot_pca(X_source, y_source, X_target, y_target, name, dir):
    X_source = X_source.detach().cpu()
    X_target = X_target.detach().cpu()
    y_source = y_source.cpu()
    y_target = y_target.cpu()

    num_class = torch.unique(y_source)
    pca = PCA(n_components=2)
    pca.fit(X_source)
    X_source_pca = pca.transform(X_source)
    X_target_pca = pca.transform(X_target)

    for i in num_class:
        i = i.item()
        plt.scatter(X_target_pca[y_target == i][:, 0], X_target_pca[y_target == i][:, 1], label='Class_' + str(i) + '_tgt')
        plt.scatter(X_source_pca[y_source == i][:, 0], X_source_pca[y_source == i][:, 1], label='Class_' + str(i) + '_src')

    plt.legend()
    plt.title(name)
    filename = dir + "/" + name + ".png"
    plt.savefig(filename)
    plt.clf()

def plot_tsne(X_source, y_source, X_target, y_target, name, dir):
    X_source = X_source.detach().cpu()
    X_target = X_target.detach().cpu()
    y_source = y_source.cpu()
    y_target = y_target.cpu()

    num_class = torch.unique(y_source)
    X_all = np.concatenate((X_source, X_target), 0)
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    X_all_tsne = tsne.fit_transform(X_all)
    X_source_tsne = X_all_tsne[0:len(X_source), :]
    X_target_tsne = X_all_tsne[len(X_source):, :]

    for i in num_class:
        i = i.item()
        plt.scatter(X_target_tsne[y_target == i][:, 0], X_target_tsne[y_target == i][:, 1], label='Class_' + str(i) + '_tgt')
        plt.scatter(X_source_tsne[y_source == i][:, 0], X_source_tsne[y_source == i][:, 1], label='Class_' + str(i) + '_src')

    plt.legend()
    plt.title(name)
    filename = dir + "/" + name + ".png"
    plt.savefig(filename)
    plt.clf()

def init_visualization(source_dataset, target_dataset, masked_training, phase, input_dim, dir, plt):
    source_x = None
    source_y = None
    target_x = None
    target_y = None

    for (graph_src, graph_tgt) in zip(source_dataset, target_dataset):
        if phase == "testing" and masked_training == 3:
            if masked_training == 0 or masked_training == 3:
                src_mask = graph_src.source_testing_mask
                tgt_mask = graph_tgt.target_testing_mask
            else:
                src_mask = graph_src.testing_mask
                tgt_mask = graph_tgt.testing_mask
        else:
            if masked_training == 0 or masked_training == 3:
                src_mask = graph_src.source_training_mask
                tgt_mask = graph_tgt.target_training_mask
            else:
                src_mask = graph_src.training_mask
                tgt_mask = graph_tgt.training_mask

        if source_x is None:
            source_x = graph_src.x[src_mask][:, 0:input_dim]
            source_y = graph_src.y[src_mask]
            target_x = graph_tgt.x[tgt_mask][:, 0:input_dim]
            target_y = graph_tgt.y[tgt_mask]
        else:
            source_x = torch.cat((source_x, graph_src.x[src_mask][:, 0:input_dim]), dim=0)
            source_y = torch.cat((source_y, graph_src.y[src_mask]), dim=0)
            target_x = torch.cat((target_x, graph_tgt.x[tgt_mask][:, 0:input_dim]), dim=0)
            target_y = torch.cat((target_y, graph_tgt.y[tgt_mask]), dim=0)
    plot_dim = source_x.size(1)
    if plot_dim == 2:
        plot_scatter(source_x, source_y, target_x, target_y, "original_" + phase, dir)
    else:
        if plt == "pca":
            plot_pca(source_x, source_y, target_x, target_y, "original_" + phase, dir)
        else:
            plot_tsne(source_x, source_y, target_x, target_y, "original_" + phase, dir)

def visual_embed(source_embed, target_embed, plt):
    names = ["GNN", "after_adv"]
    for i in range(len(source_embed)):
        source_embed_cur = source_embed[i]
        target_embed_cur = target_embed[i]
        source_embed_cur_train = source_embed_train[i]
        target_embed_cur_train = target_embed_train[i]
        name = names[i]
        plot_dim = source_embed_cur.size(1)
        if plot_dim == 2 or plot_dim == 3:
            utils.plot_scatter(source_embed_cur_train, source_label_train, target_embed_cur_train,
                               target_label_train, "Train_embedding_" + name + str(epoch), path)
            utils.plot_scatter(source_embed_cur, source_label, target_embed_cur, target_label,
                               'Test_embedding_' + name + str(epoch), path)
        elif plt == 'pca':
            utils.plot_pca(source_embed_cur_train, source_label_train, target_embed_cur_train,
                           target_label_train, "Train_embedding_" + name + str(epoch), path)
            utils.plot_pca(source_embed_cur, source_label, target_embed_cur, target_label,
                           "Test_embedding_" + name + str(epoch), path)
        else:
            utils.plot_tsne(source_embed_cur_train, source_label_train, target_embed_cur_train, target_label_train,
                            "Train_embedding_" + name + str(epoch), path)
            utils.plot_tsne(source_embed_cur, source_label, target_embed_cur, target_label,
                            "Test_embedding_" + name + str(epoch), path)

def plot_loss(epochs_train, loss_src_train, loss_src_valid, loss_src_test, loss_tgt_valid, loss_tgt_test, filename):
    plt.plot(epochs_train, loss_src_train, label='source_train')
    plt.plot(epochs_train, loss_src_valid, label='source_valid')
    plt.plot(epochs_train, loss_src_test, label='source_test')
    plt.plot(epochs_train, loss_tgt_valid, label='target_valid')
    plt.plot(epochs_train, loss_tgt_test, label='target_test')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('classification loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_dc_loss(epochs_train, loss_dc_src, loss_dc_tgt, filename):
    plt.plot(epochs_train, loss_dc_src, label='source')
    plt.plot(epochs_train, loss_dc_tgt, label='target')
    plt.xlabel('Epochs')
    plt.ylabel('loss_dc')
    plt.title('adversarial loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_str_diff(epochs, str_diff, filename):
    plt.plot(epochs, str_diff, label='str_diff')
    plt.xlabel('Epochs')
    plt.ylabel('str_diff with truth')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def get_scores(pred, label):
    auc_score = get_auc_score(pred, label)
    acc_score = get_acc_score(pred, label)
    print("auc score: " + str(auc_score))
    print("acc score: " + str(acc_score))
    print()

    return acc_score, auc_score

def cal_str_dif(pred_mtx, true_mtx):
    cls1_diff = torch.abs(pred_mtx - true_mtx)
    cls0_diff = torch.abs((1 - pred_mtx) - (1 - true_mtx))
    return 0.5 * torch.sum(cls0_diff) + 0.5 * torch.sum(cls1_diff)

def cal_str_dif_rel(pred_mtx, true_mtx):
    cls1_diff = torch.abs(pred_mtx - true_mtx)
    cls0_diff = torch.abs((1 - pred_mtx) - (1 - true_mtx))
    abs_diff = 0.5 * cls0_diff + 0.5 * cls1_diff
    rel_diff_1 = abs_diff / true_mtx
    rel_diff_2 = abs_diff / pred_mtx
    rel_diff = 0.5 * rel_diff_1 + 0.5 * rel_diff_2
    rel_diff[torch.isinf(rel_diff_1)] = rel_diff_2[torch.isinf(rel_diff_1)]
    rel_diff[torch.isinf(rel_diff_2)] = rel_diff_1[torch.isinf(rel_diff_2)]
    rel_diff[torch.isnan(rel_diff)] = 0

    num = true_mtx.size(0) * true_mtx.size(1)
    return torch.sum(abs_diff) / num, torch.sum(rel_diff) / num

def cal_str_dif_log(pred_mtx, true_mtx):
    ratio = true_mtx / pred_mtx
    ratio[torch.nonzero(ratio == 0)] = 1
    ratio[torch.isinf(ratio)] = 1
    log_matrix = torch.abs(torch.log(ratio))
    num = true_mtx.size(0) * true_mtx.size(1)
    return torch.sum(log_matrix) / num

def cal_str_diff_ratio(pred_mtx, true_mtx):
    intra_prob_pred = torch.diagonal(pred_mtx, 0).repeat_interleave(pred_mtx.size(1)).view(-1, pred_mtx.size(1))
    intra_prob_true = torch.diagonal(true_mtx, 0).repeat_interleave(true_mtx.size(1)).view(-1, true_mtx.size(1))
    pred_ratio = torch.div(pred_mtx, intra_prob_pred)
    true_ratio = torch.div(true_mtx, intra_prob_true)
    pred_ratio[torch.isnan(pred_ratio)] = 1
    true_ratio[torch.isnan(true_ratio)] = 1
    pred_ratio[torch.isinf(pred_ratio)] = pred_mtx[torch.isinf(pred_ratio)]
    true_ratio[torch.isinf(true_ratio)] = true_mtx[torch.isinf(true_ratio)]

    ratio_diff = torch.div(pred_ratio, true_ratio)
    ratio_diff[torch.isnan(ratio_diff)] = 1
    ratio_diff[torch.isinf(ratio_diff)] = 1

    num = true_mtx.size(0) * true_mtx.size(1) - pred_mtx.size(0)
    return (torch.sum(ratio_diff) - torch.sum(torch.diagonal(ratio_diff))) / num
    
