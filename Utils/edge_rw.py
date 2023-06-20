import scipy.sparse as sp
import torch
import numpy as np
#from pre_data import datasets
import math
import sys
sys.path.append('..')
from Utils import utils
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_edge_prob(src_graph, tgt_graph, dataset, domain):
    if dataset == 'SBM' or dataset == 'dblp_acm':
        src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)
    elif dataset == 'cora' or dataset == 'arxiv':
        if domain == 'word' or domain == 'degree':
            src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob_same(src_graph, tgt_graph)
        else:
            src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)
    else:
        if isinstance(src_graph, list):
            src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob_multi(src_graph, tgt_graph)
        else:
            src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)

    return src_edge_prob, tgt_edge_prob, tgt_true_edge_prob

def cal_edge_prob_same(src_graph, tgt_graph):
    # here src_graph and tgt_graph is the same
    graph = src_graph
    adj = graph.adj.T
    num_nodes = graph.num_nodes
    num_nodes_src = len(graph.source_mask)
    num_nodes_tgt = len(graph.target_mask)
    label_pred = torch.zeros(num_nodes, dtype=torch.long).to(device)
    label_pred[src_graph.source_mask] = src_graph.y[src_graph.source_mask]
    label_pred[src_graph.target_mask] = src_graph.y_hat[src_graph.target_mask]
    num_class = src_graph.num_classes

    graph_label_one_hot = sp.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), label_pred.cpu().numpy())),
                                      shape=(num_nodes, num_class))
    src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (graph.source_mask, graph.y[graph.source_mask].cpu().numpy())),
                                     shape=(num_nodes, num_class))
    tgt_pred_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (graph.target_mask, graph.y_hat[graph.target_mask].cpu().numpy())),
                                     shape=(num_nodes, num_class))
    tgt_label_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (graph.target_mask, graph.y[graph.target_mask].cpu().numpy())),
                                     shape=(num_nodes, num_class))

    src_node_num = src_label_one_hot.sum(axis=0).T * graph_label_one_hot.sum(axis=0)
    tgt_pred_node_sum = tgt_pred_one_hot.sum(axis=0).T * graph_label_one_hot.sum(axis=0)
    tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * graph_label_one_hot.sum(axis=0)
    #print(src_node_num)
    #print(tgt_pred_node_sum)
    #print(tgt_node_sum)

    src_num_edge = (src_label_one_hot.T * adj * graph_label_one_hot)
    tgt_pred_num_edge = (tgt_pred_one_hot.T * adj * graph_label_one_hot)
    tgt_true_num_edge = (tgt_label_one_hot.T * adj * graph_label_one_hot)
    #print(src_num_edge)
    #print(tgt_pred_num_edge)
    #print(tgt_true_num_edge)

    src_edge_prob = src_num_edge / src_node_num
    tgt_edge_prob = tgt_pred_num_edge / tgt_pred_node_sum
    tgt_true_edge_prob = tgt_true_num_edge / tgt_node_sum

    src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
    tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))
    tgt_true_edge_prob = torch.from_numpy(np.array(tgt_true_edge_prob))

    return src_edge_prob, tgt_edge_prob, tgt_true_edge_prob

def cal_edge_prob_multi(src_graphs, tgt_graphs):
    num_class = src_graphs[0].num_classes
    num_graphs = len(src_graphs)
    src_edge_prob = torch.zeros(num_class, num_class)
    tgt_edge_prob = torch.zeros(num_class, num_class)
    tgt_true_edge_prob = torch.zeros(num_class, num_class)

    for (src_graph, tgt_graph) in zip(src_graphs, tgt_graphs):
        src_edge_prob_i, tgt_edge_prob_i, tgt_true_edge_prob_i = cal_edge_prob_sep(src_graph, tgt_graph)
        src_edge_prob = torch.add(src_edge_prob, src_edge_prob_i)
        tgt_edge_prob = torch.add(tgt_edge_prob, tgt_edge_prob_i)
        tgt_true_edge_prob = torch.add(tgt_true_edge_prob, tgt_true_edge_prob_i)
    
    src_edge_prob = torch.div(src_edge_prob, num_graphs)
    tgt_edge_prob = torch.div(tgt_edge_prob, num_graphs)
    tgt_true_edge_prob = torch.div(tgt_true_edge_prob, num_graphs)
    return src_edge_prob, tgt_edge_prob, tgt_true_edge_prob

def cal_edge_prob_sep(src_graph, tgt_graph):
    src_adj = src_graph.adj.T
    tgt_adj = tgt_graph.adj.T
    num_nodes_src = src_graph.num_nodes
    num_nodes_tgt = tgt_graph.num_nodes
    src_label = src_graph.y
    tgt_pred = tgt_graph.y_hat
    tgt_label = tgt_graph.y
    num_class = src_graph.num_classes

    src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (np.arange(num_nodes_src), src_label.cpu().numpy())),
                                      shape=(src_graph.num_nodes, num_class))
    tgt_pred_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_pred.cpu().numpy())),
                                     shape=(tgt_graph.num_nodes, num_class))
    tgt_label_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_label.cpu().numpy())),
                                      shape=(tgt_graph.num_nodes, num_class))

    src_node_num = src_label_one_hot.sum(axis=0).T * src_label_one_hot.sum(axis=0)
    tgt_pred_node_sum = tgt_pred_one_hot.sum(axis=0).T * tgt_pred_one_hot.sum(axis=0)
    tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * tgt_label_one_hot.sum(axis=0)
    #print(src_node_num)
    #print(tgt_pred_node_sum)
    #print(tgt_node_sum)

    src_num_edge = (src_label_one_hot.T * src_adj * src_label_one_hot)
    tgt_pred_num_edge = (tgt_pred_one_hot.T * tgt_adj * tgt_pred_one_hot)
    tgt_true_num_edge = (tgt_label_one_hot.T * tgt_adj * tgt_label_one_hot)
    #print(src_num_edge)
    #print(tgt_pred_num_edge)
    #print(tgt_true_num_edge)


    src_edge_prob = src_num_edge / src_node_num
    tgt_edge_prob = tgt_pred_num_edge / tgt_pred_node_sum
    tgt_true_edge_prob = tgt_true_num_edge / tgt_node_sum

    src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
    tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))
    tgt_true_edge_prob = torch.from_numpy(np.array(tgt_true_edge_prob))

    return src_edge_prob, tgt_edge_prob, tgt_true_edge_prob

def calculate_reweight(src_graph, tgt_graph, dataset, domain):
    num_nodes = src_graph.num_nodes
    src_edge_index = src_graph.edge_index
    num_nodes_src = len(src_graph.source_mask)
    label_pred = torch.zeros(num_nodes, dtype=torch.long).to(device)
    label_pred[src_graph.source_mask] = src_graph.y[src_graph.source_mask]
    label_pred[src_graph.target_mask] = src_graph.y_hat[src_graph.target_mask]
    num_class = src_graph.num_classes

    graph_label_one_hot = sp.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), label_pred.cpu().numpy())),
                                      shape=(num_nodes, num_class))
    src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (src_graph.source_mask, src_graph.y[src_graph.source_mask].cpu().numpy())),
                                    shape=(num_nodes, num_class))
    
    src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob(src_graph, tgt_graph, dataset, domain)

    reweight_matrix = torch.div(tgt_edge_prob, src_edge_prob)
    reweight_matrix[torch.isinf(reweight_matrix)] = 1
    reweight_matrix[torch.isnan(reweight_matrix)] = 1

    print(src_edge_prob)
    print(tgt_edge_prob)
    print(reweight_matrix)
    edge_weight = torch.ones(src_graph.num_edges).to(device)

    for i in range(num_class):
        for j in range(num_class):
            idx = np.intersect1d(np.where(np.in1d(src_edge_index[0].cpu().numpy(), graph_label_one_hot.getcol(j).nonzero()[0]))[0],
                                 np.where(np.in1d(src_edge_index[1].cpu().numpy(), src_label_one_hot.getcol(i).nonzero()[0]))[0])
            edge_weight[idx] = reweight_matrix[i][j].item()
    src_graph.edge_weight = edge_weight
    print(edge_weight)
    print("rw done")

def calculate_reweight_multi(src_graphs, tgt_graphs, dataset, domain):
    src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob(src_graphs, tgt_graphs, dataset, domain)
    reweight_matrix = torch.div(tgt_edge_prob, src_edge_prob)
    reweight_matrix[torch.isinf(reweight_matrix)] = 1
    reweight_matrix[torch.isnan(reweight_matrix)] = 1

    print(src_edge_prob)
    print(tgt_edge_prob)
    print(reweight_matrix)

    if isinstance(src_graphs, list) == False:
        src_graphs = [src_graphs]
        tgt_graphs = [tgt_graphs]

    for (src_graph, tgt_graph) in zip(src_graphs, tgt_graphs):
        num_nodes = src_graph.num_nodes
        src_edge_index = src_graph.edge_index
        num_class = src_graph.num_classes

        src_label_one_hot = sp.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), src_graph.y.cpu().numpy())),
                                        shape=(num_nodes, num_class))
        
        edge_weight = torch.ones(src_graph.num_edges).to(device)

        # think of way to speed up
        # rw matrix i,j means j->i
        for i in range(num_class):
            for j in range(num_class):
                idx = np.intersect1d(np.where(np.in1d(src_edge_index[0].cpu().numpy(), src_label_one_hot.getcol(j).nonzero()[0]))[0],
                                    np.where(np.in1d(src_edge_index[1].cpu().numpy(), src_label_one_hot.getcol(i).nonzero()[0]))[0])
                edge_weight[idx] = reweight_matrix[i][j].item()
        src_graph.edge_weight = edge_weight
        #print(edge_weight)
    print("rw done")

def calculate_str_diff(src_graph, tgt_graph, dataset, domain):
    src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = cal_edge_prob(src_graph, tgt_graph, dataset, domain)
    tgt_diff_abs, tgt_diff_rel = utils.cal_str_dif_rel(tgt_edge_prob, tgt_true_edge_prob)
    src_tgt_diff_abs, src_tgt_diff_rel = utils.cal_str_dif_rel(tgt_edge_prob, src_edge_prob)
    ratio_diff = utils.cal_str_diff_ratio(tgt_edge_prob, src_edge_prob)
    log_diff = utils.cal_str_dif_log(tgt_edge_prob, src_edge_prob)

    print("str shift")
    print(src_tgt_diff_abs)
    print(src_tgt_diff_rel)
    print(ratio_diff)
    print(log_diff)

    return [src_tgt_diff_abs, src_tgt_diff_rel, ratio_diff], tgt_diff_abs

"""
def main():
    src_dataset = datasets.scbm_multi(1000, 0.8, 0.2, 0.02)
    tgt_dataset = datasets.scbm_multi(1000, 0.8, 0.2, 0.16)
    calculate_reweight(src_dataset, tgt_dataset, "SBM")

if __name__ == '__main__':
    main()
"""
