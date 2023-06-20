import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
import torch
import random
from torch_geometric.data import Data
import pickle
from scipy.spatial import distance
import h5py
from scipy import stats
import copy
from os import path
from sklearn import preprocessing
import scipy.sparse as sp
import uproot
import torch.nn.functional as F
from argparse import ArgumentParser


def edges_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index


def cal_Median_LeftRMS(x):
    """
    Given on 1d np array x, return the median and the left RMS
    """
    median = np.median(x)
    x_diff = x - median
    # only look at differences on the left side of median
    x_diffLeft = x_diff[x_diff < 0]
    rmsLeft = np.sqrt(np.sum(x_diffLeft ** 2) / x_diffLeft.shape[0])
    return median, rmsLeft


def buildConnections(eta, phi):
    """
    build the Graph based on the deltaEta and deltaPhi of input particles
    """
    phi = phi.reshape(-1, 1)
    eta = eta.reshape(-1, 1)
    dist_phi = distance.cdist(phi, phi, 'cityblock')
    indices = np.where(dist_phi > pi)
    temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
    dist_phi[indices] = dist_phi[indices] - temp
    dist_eta = distance.cdist(eta, eta, 'cityblock')
    dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
    edge_source = np.where((dist < 0.4) & (dist != 0))[0]
    edge_target = np.where((dist < 0.4) & (dist != 0))[1]
    return edge_source, edge_target


def prepare_dataset(num_event, args, datadir):
    balanced = args.balanced
    edge_feature = args.edge_feature

    print("here")
    features = [
        'PF/PF.PT', 'PF/PF.Eta', 'PF/PF.Phi', 'PF/PF.Mass',
        'PF/PF.Charge', 'PF/PF.PdgID', 'PF/PF.IsRecoPU',
        'PF/PF.IsPU'
    ]
    tree = uproot.open(datadir)["Delphes"]
    pfcands = tree.arrays(features, entry_start=0, entry_stop=0 + num_event)

    data_list = []
    feature_events = []
    charge_events = []
    label_events = []
    edge_events = []
    nparticles = []
    nChg = []
    nNeu = []
    nChg_LV = []
    nChg_PU = []
    nNeu_LV = []
    nNeu_PU = []
    pdgid_0_events = []
    pdgid_11_events = []
    pdgid_13_events = []
    pdgid_22_events = []
    pdgid_211_events = []
    pt_events = []
    eta_events = []
    phi_events = []
    mass_events = []
    pt_avg_events = []
    eta_avg_events = []
    phi_avg_events = []
    for i in range(num_event):
        if i % 1 == 0:
            print("processed {} events".format(i))
        event = pfcands[:][i]

        isCentral = (abs(event['PF/PF.Eta']) < 2.5)
        event = event[isCentral]
        pt_cut = event['PF/PF.PT'] > 0.5
        event = event[pt_cut]
        isChg = abs(event['PF/PF.Charge']) != 0
        isChgLV = isChg & (event['PF/PF.IsPU'] == 0)
        isChgPU = isChg & (event['PF/PF.IsPU'] == 1)

        isPho = (abs(event['PF/PF.PdgID']) == 22)
        isPhoLV = isPho & (event['PF/PF.IsPU'] == 0)
        isPhoPU = isPho & (event['PF/PF.IsPU'] == 1)

        isNeuH = (abs(event['PF/PF.PdgID']) == 0)
        isNeu = abs(event['PF/PF.Charge']) == 0
        isNeuLV = isNeu & (event['PF/PF.IsPU'] == 0)
        isNeuPU = isNeu & (event['PF/PF.IsPU'] == 1)

        charge_num = np.sum(np.array(isChg).astype(int))
        neutral_num = np.sum(np.array(isNeu).astype(int))
        Chg_LV = np.sum(np.array(isChgLV).astype(int))
        Chg_PU = np.sum(np.array(isChgPU).astype(int))
        Neu_LV = np.sum(np.array(isNeuLV).astype(int))
        Neu_PU = np.sum(np.array(isNeuPU).astype(int))
        nChg.append(charge_num)
        nNeu.append(neutral_num)
        nChg_LV.append(Chg_LV)
        nChg_PU.append(Chg_PU)
        nNeu_LV.append(Neu_LV)
        nNeu_PU.append(Neu_PU)

        split_idx = int(0.7 * charge_num)
        num_particle = len(isChg)
        nparticles.append(num_particle)

        # calculate deltaR
        eta = np.array(event['PF/PF.Eta'])
        phi = np.array(event['PF/PF.Phi'])
        edge_source, edge_target = buildConnections(eta, phi)
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        edge_events.append(edge_index)
        print("done!!")

        adj, _ = edges_to_adj(edge_source, edge_target, num_particle)

        # node features
        pt = np.array(event['PF/PF.PT'])
        mass = np.array(event['PF/PF.Mass'])
        pdgid_0 = (abs(event['PF/PF.PdgID']) == 0)
        pdgid_11 = (abs(event['PF/PF.PdgID']) == 11)
        pdgid_13 = (abs(event['PF/PF.PdgID']) == 13)
        pdgid_22 = (abs(event['PF/PF.PdgID']) == 22)
        pdgid_211 = (abs(event['PF/PF.PdgID']) == 211)

        isReco = np.array(event['PF/PF.IsRecoPU'])
        chg = np.array(event['PF/PF.Charge'])
        label = np.array(event['PF/PF.IsPU'])
        pdgID = np.array(event['PF/PF.PdgID'])

        # truth label
        label = torch.from_numpy(label == 0)
        label = label.type(torch.long)

        charge_events.append(chg)
        label_events.append(label)
        name = [isChgLV, isChgPU, isNeuLV, isNeuPU]
        if i == 0:
            for node_i in range(4):
                pt_events.append(pt[name[node_i]])
                eta_events.append(eta[name[node_i]])
                phi_events.append(phi[name[node_i]])
                mass_events.append(mass[name[node_i]])
                pdgid_0_events.append([np.sum(np.array(pdgid_0 & name[node_i]).astype(int))])
                pdgid_11_events.append([np.sum(np.array(pdgid_11 & name[node_i]).astype(int))])
                pdgid_13_events.append([np.sum(np.array(pdgid_13 & name[node_i]).astype(int))])
                pdgid_22_events.append([np.sum(np.array(pdgid_22 & name[node_i]).astype(int))])
                pdgid_211_events.append([np.sum(np.array(pdgid_211 & name[node_i]).astype(int))])
                pt_avg_events.append([np.average(np.array(pt[name[node_i]]))])
                eta_avg_events.append([np.average(np.array(abs(eta[name[node_i]])))])
                phi_avg_events.append([np.average(np.array(abs(phi[name[node_i]])))])

        else:
            for node_i in range(4):
                pt_events[node_i] = np.concatenate((pt_events[node_i], pt[name[node_i]]))
                eta_events[node_i] = np.concatenate((eta_events[node_i], eta[name[node_i]]))
                phi_events[node_i] = np.concatenate((phi_events[node_i], phi[name[node_i]]))
                mass_events[node_i] = np.concatenate((mass_events[node_i], mass[name[node_i]]))
                pdgid_0_events[node_i].append(np.sum(np.array(pdgid_0 & name[node_i]).astype(int)))
                pdgid_11_events[node_i].append(np.sum(np.array(pdgid_11 & name[node_i]).astype(int)))
                pdgid_13_events[node_i].append(np.sum(np.array(pdgid_13 & name[node_i]).astype(int)))
                pdgid_22_events[node_i].append(np.sum(np.array(pdgid_22 & name[node_i]).astype(int)))
                pdgid_211_events[node_i].append(np.sum(np.array(pdgid_211 & name[node_i]).astype(int)))
                pt_avg_events[node_i].append(np.average(np.array(pt[name[node_i]])))
                eta_avg_events[node_i].append(np.average(np.array(abs(eta[name[node_i]]))))
                phi_avg_events[node_i].append(np.average(np.array(abs(phi[name[node_i]]))))

        # ffm for eta and pt
        phi = torch.from_numpy(phi)
        eta = torch.from_numpy(eta)
        pt = torch.from_numpy(pt)
        B_eta = torch.randint(0, 10, (1, 5), dtype=torch.float)
        B_pt = torch.randint(0, 10, (1, 5), dtype=torch.float)
        alpha_eta = 1
        alpha_pt = 1
        eta_ffm = (2 * pi * alpha_eta * eta).view(-1, 1) @ B_eta
        eta_ffm = torch.cat((torch.sin(eta_ffm), torch.cos(eta_ffm)), dim=1)
        pt_ffm = (2 * pi * alpha_pt * pt).view(-1, 1) @ B_pt
        pt_ffm = torch.cat((torch.sin(pt_ffm), torch.cos(pt_ffm)), dim=1)
        node_features = np.concatenate((eta_ffm, pt_ffm), axis=1)

        # no charge information as full simulation
        # node_features = np.concatenate((eta.reshape(-1, 1), pt.reshape(-1, 1)), axis=1)


        # one hot encoding of label
        label_copy = copy.deepcopy(label)
        label_copy[isNeu] = 2
        label_onehot = F.one_hot(label_copy)

        # one hot encoding of pdgID
        pdgID_copy = copy.deepcopy(pdgID)
        pdgID_copy[pdgid_0] = 0
        pdgID_copy[pdgid_11] = 1
        pdgID_copy[pdgid_13] = 2
        pdgID_copy[pdgid_22] = 3
        pdgID_copy[pdgid_211] = 4
        pdgID_onehot = F.one_hot(torch.from_numpy(pdgID_copy).type(torch.long))

        if edge_feature:
            node_features = np.concatenate((node_features, pdgID_onehot, label_onehot, eta.view(-1, 1), phi.view(-1, 1)), axis=1)
        else:
            node_features = np.concatenate((node_features, pdgID_onehot, label_onehot), axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        feature_events.append(node_features)
        graph = Data(x=node_features, edge_index=edge_index, y=label)
        graph.adj = adj
        graph.y_hat = label
        graph.edge_weight = torch.ones(graph.num_edges)
        #Neu_indices = np.where(np.array(isNeu) == True)[0]
        #Chg_indices = np.where(np.array(isChg) == True)[0]
        graph.Charge_LV = np.where(np.array(isChgLV) == True)[0]
        graph.Charge_PU = np.where(np.array(isChgPU) == True)[0]
        #np.random.shuffle(Neu_indices)
        #np.random.shuffle(Chg_indices)
        if len(graph.Charge_LV) < 10:
            continue
    
        Neu_LV_indices = np.where(np.array(isNeuLV) == True)[0]
        Neu_PU_indices = np.where(np.array(isNeuPU) == True)[0]
        np.random.shuffle(Neu_LV_indices)
        np.random.shuffle(Neu_PU_indices)

        if len(Neu_LV_indices) < 2:
            continue
        num_training_LV = len(Neu_LV_indices)
        num_training_PU = len(Neu_PU_indices)

        if balanced:
            num_training_PU = num_training_LV

        # training and testing on all neutral particles
        training_mask = np.concatenate((Neu_LV_indices[0:num_training_LV],
                                        Neu_PU_indices[0:num_training_PU]))
        np.random.shuffle(training_mask)
        graph.training_mask = training_mask

        graph.num_classes = 2
        print("done!!!")
        data_list.append(graph)

    # plot the distribution
    """
    plt.figure(figsize=(8, 6))
    plt.hist(nChg_LV, histtype='step', color='black', label='Charge_LV')
    plt.hist(nChg_PU, histtype='step', color='red', label='Charge_PU')
    plt.hist(nNeu_LV, histtype='step', color='green', label='Neutral_LV')
    plt.hist(nNeu_PU, histtype='step', color='blue', label='Neutral_PU')
    plt.xlabel('nParticle')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(nparticles, histtype='step', color='black', label='total')
    plt.hist(nNeu, histtype='step', color='red', label='Neutral')
    plt.hist(nChg, histtype='step', color='green', label='Charge')
    plt.xlabel('nParticle')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pt_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pt_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pt_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pt_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pt')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pt_avg_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pt_avg_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pt_avg_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pt_avg_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pt_event')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(eta_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(eta_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(eta_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(eta_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('Eta')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(eta_avg_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(eta_avg_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(eta_avg_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(eta_avg_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('Eta_event')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(phi_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(phi_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(phi_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(phi_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('Phi')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(phi_avg_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(phi_avg_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(phi_avg_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(phi_avg_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('Phi_event')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(mass_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(mass_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(mass_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(mass_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('Mass')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pdgid_0_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pdgid_0_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pdgid_0_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pdgid_0_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pdgId_0')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pdgid_11_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pdgid_11_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pdgid_11_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pdgid_11_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pdgid_11')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pdgid_13_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pdgid_13_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pdgid_13_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pdgid_13_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pdgid_13')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pdgid_22_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pdgid_22_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pdgid_22_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pdgid_22_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pdgid_22')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(pdgid_211_events[0], histtype='step', color='black', label='Chg_LV')
    plt.hist(pdgid_211_events[1], histtype='step', color='red', label='Chg_PU')
    plt.hist(pdgid_211_events[2], histtype='step', color='green', label='Neu_LV')
    plt.hist(pdgid_211_events[3], histtype='step', color='blue', label='Neu_PU')
    plt.xlabel('pdgid_211')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    """
    return data_list


"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblp')
parser.add_argument("--target", type=str, default='acm')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int,default=20)
parser.add_argument("--UDAGCN", type=bool,default=True)
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--encoder_dim", type=int, default=300)
parser.add_argument("--f_dim", type=int, default=300)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--class_layers", type=int, default=2)
parser.add_argument("--dataset", type=str, default='dblp_acm')
parser.add_argument("--masked_training", type=int, default=1)
parser.add_argument("--balanced", type=bool, default=True)
parser.add_argument('--edge_feature', type=bool, default = False)


#args = parser.parse_args()
#prepare_dataset(10, True, args, "./pileup/PU10/test_gg_PU10_3K_v2.root")
"""
