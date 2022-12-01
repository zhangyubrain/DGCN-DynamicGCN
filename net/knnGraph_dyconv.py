# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:43:17 2020

@author: 99488
"""
import torch
from torch import nn
from torch_cluster import knn_graph
from torch_geometric.nn import GATConv,NNConv,EdgeConv
from net.mlp import MLP
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
from net.le_conv import LEConv
from torch_geometric.nn.pool.topk_pool import topk

class EdgConv(EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, drop=0, aggr='mean'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias, drop), aggr)

    def forward(self, x, edge_index):
        # return super(EdgConv, self).forward(x, edge_index, edge_attr)
        return super(EdgConv, self).forward(x, edge_index)

class Dilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index, edge_attr, batch=None):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]              
                edge_attr = edge_attr.view(1, -1, num)
                edge_attr = edge_attr[:, :, randnum]
                return edge_index.view(2, -1), edge_attr.view(1, -1)
            else:
                edge_index = edge_index[:, ::self.dilation]
                edge_attr = edge_attr[:, ::self.dilation]
        else:
            edge_index = edge_index[:, ::self.dilation]
        return edge_index, edge_attr
    
class DilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(k, dilation, stochastic, epsilon)
        if knn == 'matrix':
            self.knn = knn_matrix(self.k * self.dilation)
        else:
            self.knn = knn_graph

    def forward(self, x, batch):
        edge_index, edge_attr = self.knn(x, batch)
        return self._dilated(edge_index, edge_attr, batch)


class knn_matrix(nn.Module):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (num_points, num_points)
        k: int
    Returns:
        nearest neighbors: (num_points*k ,1) (num_points, k)
    """
    def __init__(self, k=16):
        super(knn_matrix, self).__init__()
        self.k = k
        # self.lin1 = torch.nn.Linear(128, 80)
        # self.lin2 = torch.nn.Linear(80, 1)

    def forward(self, x, batch):
        if batch is None:
            batch_size = 1
        else:
            batch_size = batch[-1] + 1
            
        x = x.view(batch_size, -1, x.shape[-1])
    
        neg_adj = -pairwise_distance(x)
    
        val, nn_idx = torch.topk(neg_adj, k=self.k)
        del neg_adj
    
        n_points = x.shape[1]
        start_idx = torch.arange(0, n_points*batch_size, n_points).long().view(batch_size, 1, 1)
        if x.is_cuda:
            start_idx = start_idx.cuda("cuda:0")
        nn_idx += start_idx
        del start_idx
    
        if x.is_cuda:
            torch.cuda.empty_cache()
    
        nn_idx = nn_idx.view(1, -1)
        val = val.view(1,-1).cuda("cuda:0")
        center_idx = torch.arange(0, n_points*batch_size).repeat(self.k, 1).transpose(1, 0).contiguous().view(1, -1)
        if x.is_cuda:
            center_idx = center_idx.cuda("cuda:0")
        return  torch.cat((nn_idx, center_idx), dim=0), val


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)#欧式距离的优化表达

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',
                 act='relu', norm=None, bias=True, drop=0, heads=8):
        super(GraphConv, self).__init__()
        self.drop = drop
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias, drop)
        elif conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))
            
        # self.lin1 = torch.nn.Linear(in_channels, out_channels//2)
        # self.lin2 = torch.nn.Linear(out_channels//2, 1)
        # self.gnn_score = LEConv(in_channels, 1)
    def forward(self, x, edge_index, batch = None):

        x1 = self.gconv(x, edge_index)
        # x2 = self.lin1(x1)
        # x2 = self.lin2(x2)
        # x_s = F.softmax(F.leaky_relu(x2),dim=0)
        # x_s = F.dropout(x_s,p= 0.5, training=self.training)
        # x = x1 * x_s
        # fitness = torch.sigmoid(self.gnn_score(x1, edge_index=edge_index))
        # x = x * fitness
        return x1
    
class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, drop=0, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, drop, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        edge_index,_ = dropout_adj(edge_index,p=0.2,training=self.training)
        return super(DynConv, self).forward(x, edge_index, batch), edge_index
    
class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, drop=0, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, drop, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        out = self.body(x, batch)
        return out[0] + x*self.res_scale, out[1], batch

