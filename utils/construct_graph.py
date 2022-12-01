
import os.path as osp
from os import listdir
import glob
import os

import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import scipy.io as sio
from scipy.spatial import distance
import math
def split(data, batch1, batch2):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch1)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch1[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])  

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch1[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}   
    # if data.aal_edge_index is not None:
    #     row_aal, _ = data.aal_edge_index
    #     edge_slice_aal = torch.cumsum(torch.from_numpy(np.bincount(batch2[row_aal])), 0)
    #     edge_slice_aal = torch.cat([torch.tensor([0]), edge_slice_aal]) 
    #     data.aal_edge_index -= node_slice[batch2[row_aal]].unsqueeze(0)
    #     slices['aal_edge_index'] = edge_slice_aal  
    if data.x is not None:
        slices['x'] = node_slice    
    # if data.aalx is not None:
    #     slices['aalx'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice  
    # if data.aal_edge_attr is not None:
    #     slices['aal_edge_attr'] = edge_slice_aal
    if data.y is not None:
        if data.y.size(0) == batch1.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch1[-1] + 2, dtype=torch.long)   
    if data.pos is not None:
        slices['pos'] = node_slice
    if data.pcd is not None:
        slices['pcd'] = torch.arange(0, batch1[-1] + 2, dtype=torch.long)    
    if data.clinic_score is not None:
        slices['clinic_score'] = torch.arange(0, batch1[-1] + 2, dtype=torch.long)

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

def get_site_idx(info_list):
    site_list = [x[-1] for x in info_list]
    site_set = set(site_list)
    index_list = []
    for j in site_set:
        index = []
        for i,info in enumerate(site_list):
            if info == j:
                index.append(i)
        index_list.append(index)
            
    return index_list

def norm_site(node_fea, edge, index):
    node_array = np.array(node_fea)
    edge_array = np.array(edge)
    node = range(node_array.shape[1])

    for i in range(len(index)):
        min_node = np.min(node_array[index[i]],0)
        min_edge = np.min(edge_array[index[i]],0)
        max_node = np.max(node_array[index[i]],0)
        max_edge = np.max(edge_array[index[i]],0)
        avr_node = np.mean(node_array[index[i]],axis=0)
        avr_edge = np.mean(edge_array[index[i]],axis=0)
        std_node = np.std(node_array[index[i]],axis=0)
        std_edge = np.std(edge_array[index[i]],axis=0)
        #stardadization
        node_array[index[i]] = (node_array[index[i]]-avr_node)/std_node
        edge_array[index[i]] = (edge_array[index[i]]-avr_edge)/std_edge
        #norm
        range_node = (max_node-min_node) 
        node_array[index[i]] = t = (node_array[index[i]]-min_node)/range_node
        edge_array[index[i]] = (edge_array[index[i]]-min_edge)/(max_edge-min_edge)
        
    for j in range(len(node_array)):
        node_array[j][node,node] = 0
    return node_array, edge_array

def read_data(data_dir):
    onlyfile = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    # onlyfiles.sort()
    onlyfiles = [data_dir+'/'+f for f in onlyfile]
    batch = []
    y_list = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    
    import timeit
    start = timeit.default_timer()
    
    # res = read_sigle_data('correlation' ,'partial_corr', onlyfiles[0])

    import sys
    BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR, 'utils'))   
    # BASE_DIR = os.path.dirname(os.path.abspath(data_dir))
    data_dir2 = r'H:\PHD\learning\research\PRGNN_fMRI-myModify\data\new\test\raw_aal_remove2'
    onlyfile2 = [f for f in listdir(data_dir2) if osp.isfile(osp.join(data_dir2, f))]    
    onlyfiles2 = [data_dir2+'/'+f for f in onlyfile2]
    
    
    # parallar computing
    pool = multiprocessing.Pool(processes=6)
    func = partial(read_sigle_data, 'correlation' ,'correlation')

        
    res = list(pool.map(func, onlyfiles))  
    res2 = list(pool.map(func, onlyfiles2))    
    res.extend(res2)
    
    pool.close()
    pool.join()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
    # tem = []
    idx = []
    for i,j in enumerate(res):
        # tem.append(j[-1])
        if j:
            if math.isnan(j[5][-1]) == False and math.isnan(j[5][-2]) == False and math.isnan(j[5][-3]) == False:
                idx.append(i)
    
    info = [res[i] for i in idx]    
    shf_data, shf_batch = construct_data(info)

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))
        
    # index_list = get_site_idx(res)
    # node_array, edge_array = norm_site(att_list, edge_att_list, index_list)#对地点norm
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    
    #edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    #data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch)
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    # shf_data.aalx = att_torch
    # shf_data.aal_edge_index = edge_index_torch
    # shf_data.aal_edge_attr = edge_att_torch
    
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch)    
    batch_torch = None

    data, slices = split(shf_data, shf_batch, batch_torch)
    
    return data, slices

def construct_data (data):
    y_list = []
    pseudo = []   
    batch = []
    edge_att_list, edge_index_list,att_list, pcd, clinic_score = [], [], [], [], []
    for j in range(len(data)):
        edge_att_list.append(data[j][0])
        edge_index_list.append(data[j][1]+j*data[j][4])
        att_list.append(data[j][2])
        y_list.append(data[j][3])       
        batch.append([j]*data[j][4])
        pseudo.append(np.diag(np.ones(data[j][4])))
        # pcd.append(data[j][5][1:])
        pcd.append(data[j][5])
        clinic_score.append(data[j][6])

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list).astype(float)
    
    # y_1 = np.where(y_arr==1)[0]
    # y_0 = np.where(y_arr==0)[0]
    # y_arr[y_1] = y_arr[y_1]/(len(y_arr)-len(y_1))
    # y_arr[y_0] = -1
    # y_arr[y_0] = y_arr[y_0]/(len(y_arr)-len(y_0))
    # y_arr = y_arr *1000
    pseudo_arr = np.concatenate(pseudo, axis=0)
    pcd_array = np.array(pcd)
    score_array = np.array(clinic_score)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd = torch.from_numpy(pcd_array).float()
    clinic_score = torch.from_numpy(score_array).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch, pcd = pcd, clinic_score = clinic_score)
    # data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch, pcd = pcd)
    return data, batch_torch


def read_sigle_data(edge, node_fea , data_dir):
# Load precomputed fMRI connectivity networks

    temp = sio.loadmat(data_dir)
    measure= float(temp['adhd_measure'])
    label = int(temp['indicator'])
    # if label != 2:
    if label!=0:
        label=1   
    # if Site != 7:#or Site == 1 or Site == 3
        # read edge and edge attribute
    # pcorr = np.abs(np.arctanh(temp[edge]))
    pcorr = np.arctanh(temp[edge])# only for visulization
    
    # only keep the top 10% edges
    th = np.percentile(pcorr.reshape(-1),80)
    pcorr[pcorr < th] = 0  # set a threshold
    num_nodes = pcorr.shape[0]

    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros((len(adj.row)))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), torch.from_numpy(edge_att).float())
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)

    att = temp[node_fea]
    
    gender = float(temp['Gender'])
    age = float(temp['Age'])
    handness = float(temp['Handedness'])
    fiq = float(temp['FIQ'])
    piq = float(temp['PIQ'])
    viq = float(temp['VIQ'])
    pcd = [measure,gender,age,handness,fiq,piq,viq]
    # if temp['adhd_measure'] == 'NaN' or temp['adhd_index'] == 'NaN':
    #     i =1
    socre = [float(temp['adhd_measure']), float(temp['adhd_index']), float(temp['Inattentive']), float(temp['Hyper_Impulsive'])]
    # socre = [float(temp['Inattentive']), float(temp['Hyper_Impulsive'])]
    return edge_att.data.numpy(),edge_index.data.numpy(),att, label, num_nodes, pcd, socre
    # else:
    #     return
    
