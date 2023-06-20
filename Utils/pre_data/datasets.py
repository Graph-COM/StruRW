from torch.utils.data import Dataset
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp

import random
import math
import torch.nn.functional as F
from sparsebm import generate_SBM_dataset
import networkx as nx
import scipy
import csv
import json
import os
import sys
from torch_geometric.io import read_txt_array

sys.path.append('..')
from Utils.pre_data import prepare_pileup
from Utils.pre_data import pre_cora
from Utils.pre_data import pre_arxiv
from Utils.pre_data import pre_arxiv_GOOD

class MyDataset(Dataset):
    def __init__(self, feature, labels, domain):
        self.labels = labels
        self.feature = feature
        self.domain = domain

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.feature[idx]
        domain = self.domain[idx]
        sample = {"feature": data, "Class": label, "Domain": domain}
        return sample


def read_edges(edge_fp):
    edges = edge_fp.readlines()
    edges = [tuple(row.split()) for row in edges]
    edges = [(float(row[0]), float(row[1])) for row in edges]
    return edges


def edges_to_adj(edges, num_node):
    edge_source = [int(i[0]) for i in edges]
    edge_target = [int(i[1]) for i in edges]
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index


def pileup(num_events, args, datadir):
    graph_list = prepare_pileup.prepare_dataset(num_events, args, datadir)
    return graph_list

def scbm_multi(num_nodes, SIGMA, p, q):
    B = [[p, q, q],
    [q, p, q],
    [q, q, p]]
    B = [[x * 0.1 for x in y] for y in B]

    n = [num_nodes, num_nodes, num_nodes]
    G = nx.stochastic_block_model(n, B)
    edge_list = list(G.edges)
    C1 = np.random.multivariate_normal(mean=[1, 0], cov=[[SIGMA ** 2, 0], [0, SIGMA ** 2]], size=num_nodes)
    C0 = np.random.multivariate_normal(mean=[-1, 0], cov=[[SIGMA ** 2, 0], [0, SIGMA ** 2]], size=num_nodes)
    C2 = np.random.multivariate_normal(mean=[3, 2], cov=[[SIGMA ** 2, 0], [0, SIGMA ** 2]], size=num_nodes)

    node_idex = np.arange(3 * num_nodes)
    features = np.zeros((3 * num_nodes, C1.shape[1]))
    label = np.zeros((3 * num_nodes))

    c0_idx = node_idex[list(G.graph['partition'][0])]
    c1_idx = node_idex[list(G.graph['partition'][1])]
    c2_idx = node_idex[list(G.graph['partition'][2])]

    features[c0_idx] = C0
    features[c1_idx] = C1
    features[c2_idx] = C2

    label[c1_idx] = 1
    label[c2_idx] = 2

    random.shuffle(c0_idx)
    random.shuffle(c1_idx)
    random.shuffle(c2_idx)

    features = torch.FloatTensor(features)
    label = torch.LongTensor(label)
    #domain = torch.FloatTensor(domain)
    idx_source_train = np.concatenate((c0_idx[:int(0.6 * len(c0_idx))],
                                 c1_idx[:int(0.6 * len(c1_idx))], c2_idx[:int(0.6 * len(c2_idx))]))
    idx_source_valid = np.concatenate((c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
                                      c1_idx[int(0.6 * len(c1_idx)) : int(0.8 * len(c1_idx))], c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]))
    idx_source_test = np.concatenate((c0_idx[int(0.8 * len(c0_idx)):],
                                c1_idx[int(0.8 * len(c1_idx)):], c2_idx[int(0.8 * len(c2_idx)):]))
    idx_target_valid = np.concatenate((c0_idx[:int(0.2 * len(c0_idx))],
                                       c1_idx[:int(0.2 * len(c1_idx))], c2_idx[:int(0.2 * len(c2_idx))]))
    idx_target_test = np.concatenate((c0_idx[int(0.2 * len(c0_idx)):],
                                c1_idx[int(0.2 * len(c1_idx)):], c2_idx[int(0.2 * len(c2_idx)):]))
    num_nodes = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)

    graph.adj = adj
    graph.y_hat = label
    graph.num_classes = 3
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph

def prepare_dblp_acm(raw_dir, name):
    docs_path = os.path.join(raw_dir, name, 'raw/{}_docs.txt'.format(name))
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    num_node = x.size(0)
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)
    y = torch.from_numpy(y).to(torch.int64)
    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = y
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph

def prepare_cora(root, domain, shift):
    dataset_obj = pre_cora.GOODCora.load(root, domain, shift)
    dataset = dataset_obj.data

    source_training_mask = dataset.train_mask
    source_validation_mask = dataset.id_val_mask
    source_testing_mask = dataset.id_test_mask
    source_mask = dataset.train_mask + dataset.id_val_mask + dataset.id_test_mask
    target_mask = dataset.val_mask + dataset.test_mask
    target_idx = (target_mask == 1).nonzero().view(-1).numpy()
    np.random.shuffle(target_idx)

    dataset.source_training_mask = (source_training_mask == 1).nonzero().view(-1).numpy()
    dataset.source_validation_mask = (source_validation_mask == 1).nonzero().view(-1).numpy()
    dataset.source_testing_mask = (source_testing_mask == 1).nonzero().view(-1).numpy()
    dataset.source_mask = (source_mask == 1).nonzero().view(-1).numpy()
    dataset.target_validation_mask = target_idx[0:int(0.2*len(target_idx))]
    dataset.target_testing_mask = target_idx[int(0.2*len(target_idx)):]
    dataset.target_mask = target_idx

    dataset.num_classes = dataset_obj.num_classes
    dataset.edge_weight = torch.ones(dataset.num_edges)
    edge_index = dataset.edge_index
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(dataset.num_nodes, dataset.num_nodes))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    dataset.adj = adj
    dataset.y_hat = dataset.y
    return dataset

def prepare_arxiv(root, years):
    if years == "degree":
        dataset_obj, _ = pre_arxiv_GOOD.GOODArxiv.load(root, years, 'covariate')
        dataset = dataset_obj.data
        graph = Data(edge_index=dataset.edge_index, x=dataset.x, y=dataset.y.view(-1))
        source_training_mask = dataset.train_mask
        source_validation_mask = dataset.id_val_mask
        source_testing_mask = dataset.id_test_mask
        source_mask = dataset.train_mask + dataset.id_val_mask + dataset.id_test_mask
        target_mask = dataset.val_mask + dataset.test_mask
        target_idx = (target_mask == 1).nonzero().view(-1).numpy()
        np.random.shuffle(target_idx)

        graph.source_training_mask = (source_training_mask == 1).nonzero().view(-1).numpy()
        graph.source_validation_mask = (source_validation_mask == 1).nonzero().view(-1).numpy()
        graph.source_testing_mask = (source_testing_mask == 1).nonzero().view(-1).numpy()
        graph.source_mask = (source_mask == 1).nonzero().view(-1).numpy()
        graph.target_validation_mask = target_idx[0:int(0.2 * len(target_idx))]
        graph.target_testing_mask = target_idx[int(0.2 * len(target_idx)):]
        graph.target_mask = target_idx

        graph.num_classes = dataset_obj.num_classes
        graph.edge_weight = torch.ones(dataset.num_edges)
        edge_index = dataset.edge_index
        data = np.ones(edge_index.size(1))
        adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                            shape=(dataset.num_nodes, dataset.num_nodes))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        graph.adj = adj
        graph.y_hat = dataset.y
        return graph
    else:
        # need to check the split of test nodes and the number of nodes in the graph
        dataset = pre_arxiv.load_nc_dataset(root, 'ogb-arxiv', years)
        idx = (dataset.test_mask == True).nonzero().view(-1).numpy()
        np.random.shuffle(idx)
        num_training = idx.shape[0]
        graph = Data(edge_index=dataset.graph['edge_index'], x=dataset.graph['node_feat'], y=dataset.label.view(-1))
        edge_index = dataset.graph['edge_index']
        data = np.ones(edge_index.size(1))
        adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                            shape=(graph.num_nodes, graph.num_nodes))
        #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        graph.adj = adj
        graph.source_training_mask = idx[0:int(0.6*num_training)]
        graph.source_validation_mask = idx[int(0.6*num_training):int(0.8*num_training)]
        graph.source_testing_mask = idx[int(0.8*num_training):]
        graph.target_validation_mask = idx[0:int(0.2*num_training)]
        graph.target_testing_mask = idx[int(0.2*num_training):]
        graph.source_mask = idx
        graph.target_mask = idx
        graph.edge_weight = torch.ones(graph.num_edges)
        graph.num_classes = dataset.num_classes
        if torch.unique(graph.y).size(0) < graph.num_classes:
            print("miss classes")
            #return 
        graph.y_hat = graph.y 
        return graph




