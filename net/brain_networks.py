import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,NNConv,EdgeConv,MessagePassing,DNAConv, GINEConv
from torch_geometric.nn.conv import GCNConv
from net.mlp import MLP

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp
from torch_geometric.utils import (add_self_loops, sort_edge_index,dropout_adj, degree, remove_self_loops, to_dense_adj, dense_to_sparse)
from torch_sparse import spspmm

from net.MyTopK import TopKPooling
from net.MySAG import SAGPooling
from net.myASAP import ASAPooling
from net.knnGraph_dyconv import DilatedKnnGraph, ResDynBlock, EdgConv
from net.score import Get_score
from net.le_conv import LEConv

class Net(torch.nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.dim1 = 12
        self.dim2 = 12
        self.dim3 = 8#改動
        # self.dim3 = 4
        head1=4
        # head2=4
        # self.ep_net = VGAE(opt.indim, self.dim1, head1, self.dim1, norm=opt.norm, drop=0.2, gae=False)

        # self.ssl_net = GNN(opt.shf_indim, self.dim1, head1, opt.k, opt.stochastic, opt.epsilon, opt.ratio,
        #                     opt.norm, opt.conv, opt.poolmethod, opt.bias, drop=0.2)        
        self.ssl_net = GNN(opt.aal_indim, self.dim1, head1, opt.k, opt.stochastic, opt.epsilon, opt.ratio,
                            opt.norm, opt.conv, opt.poolmethod, opt.bias, drop=0.3)
        
        # self.fc0 = torch.nn.Linear(opt.indim, (self.dim1*head1)*2)
        # self.bn0 = torch.nn.BatchNorm1d((self.dim1*head1)*2)
              
        self.fc1 = torch.nn.Linear(116+(116)*2, self.dim2*2)
        # self.fc1 = torch.nn.Linear((self.dim1*head1)*2, self.dim2*2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2*2)      
        self.fc2 = torch.nn.Linear(self.dim2*2+7, self.dim3)#改動
        # self.fc2 = torch.nn.Linear(self.dim2*2, self.dim3)#改動
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2)
        # self.fc_pcd = torch.nn.Linear(6, 1)
        
    
    def forward(self, x, edge_index, batch, edge_attr, pcd):
        # edge_idx, edge_raw, edge_r, edge_index, edge_attr= self.ep_net(x, edge_index, edge_attr, batch)
        # adj_r = to_dense_adj(edge_index, batch, edge_attr)#raw edge matrix
        
        x1, score = self.ssl_net(x, edge_index, batch, edge_attr, pcd)#改動
        pcd = F.normalize(pcd,dim=1)
        # # pcd_mask = F.sigmoid(self.fc_pcd(pcd))
        # # x = x1 * pcd_mask
        x = self.bn4(F.relu(self.fc1(x1)))#改動
        x = F.dropout(x, p=0.5, training=self.training)#改動
        # x = pcd#改動
        x = torch.cat([x,pcd], dim=1)#改動
        x = self.bn5(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
# F.softmax(x,-1)
        return F.softmax(x,-1), score ,x1, x

class GNN(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, indim, dim1, head1, k, stochastic, epsilon, ratio, norm, conv, poolmethod, bias, drop):
        super(GNN, self).__init__()
        self.indim = indim
        self.dim1 = dim1
        self.head1 = head1
        self.knn = DilatedKnnGraph(k, 2, stochastic, epsilon)
        self.knn2 = DilatedKnnGraph(k, 2, stochastic, epsilon)

        if conv == 'edge':
            self.conv1 = EdgConv( indim, 116, norm=norm, drop=drop)
            self.conv1_1 = EdgConv( indim, 116, norm=norm, drop=drop)
            self.conv2 = EdgConv( indim, 1, norm=norm, drop=drop)
            self.conv2_2 = EdgConv( indim, 1, norm=norm, drop=drop)
            self.conv3 = EdgConv( indim, 116, norm=norm, drop=drop)           
            self.conv3_3 = EdgConv( indim, 116, norm=norm, drop=drop)           
            
        elif conv == 'gat':
            self.conv1 = GATConv( indim, indim, bias=bias, dropout=drop)
            self.conv2 = GATConv( indim, 1, bias=bias, dropout=drop)
            self.conv3 = GATConv( indim, indim, bias=bias)
            
        elif conv == 'DNA':
            self.conv1 = EdgConv( indim, dim1*head1, norm=norm, drop=0)
            self.conv2 = DNAConv(head1*dim1, heads= head1, groups = dim1, dropout=0.)
                               
        elif conv == 'gcn':
            self.conv1 = GCNConv( indim, indim, bias=bias, dropout=drop)
            self.conv2 = GCNConv( indim, 1)
            self.conv3 = GCNConv( indim, indim)
        self.bn1 = torch.nn.BatchNorm1d(116)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.bn3 = torch.nn.BatchNorm1d(116)

        if poolmethod == 'topk':
            self.pool1_1 = TopKPooling(dim1*head1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
            self.pool1_2 = TopKPooling(dim1*head1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        elif poolmethod == 'sag':
            self.pool1_1 = SAGPooling(dim1*head1, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid) #0.4 data1 10 fold
            self.pool1_2 = SAGPooling(dim1*head1, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid) #0.4 data1 10 fold        
            
        elif poolmethod == 'ASA':
            self.pool1_1 = ASAPooling(dim1*head1, ratio=ratio, GNN=GCNConv) #0.4 data1 10 fold
            self.pool1_2 = ASAPooling(dim1*head1, ratio=ratio, GNN=GCNConv) #0.4 data1 10 fold
        else:
            self.pool1_1 =False
        
        # self.fc1 = torch.nn.Linear(indim, 1)
        # self.fc2 = torch.nn.Linear(3, 1)
    def get_2hop_idx(self, x, edge_index, batch):
               
        adj_get = to_dense_adj(edge_index, batch)#raw edge matrix

        for i in range(x.shape[0]//116):
            if i ==0 :
                edge1_2 = torch.mm(adj_get[i], adj_get[i])
                edge1_2 = edge1_2*torch.abs(adj_get[i]-1)
                edge_ = dense_to_sparse(edge1_2)
                e1_2hop = torch.cat((edge_[0][1].unsqueeze(0), edge_[0][0].unsqueeze(0)))
            else:
                edge_2 = torch.mm(adj_get[i], adj_get[i])
                edge_2 = edge_2*torch.abs(adj_get[i]-1)
                edge__ = dense_to_sparse(edge_2)
                edge__ = torch.cat((edge__[0][1].unsqueeze(0), edge__[0][0].unsqueeze(0)))
                # edge__ = dense_to_sparse(edge_2)

                e1_2hop = torch.cat((e1_2hop, edge__+116*i),1)
                
        return e1_2hop
    
    def forward(self, x, edge_index, batch, edge_attr, pcd):
        # GCN encoder
        # x = F.dropout(x,p= 0.2, training=self.training)
        edge_index, _ = dropout_adj(edge_index, edge_attr, p=0.3,training=self.training)

        e1, _ = self.knn(x, batch)#knn动态
        e1, _ = dropout_adj(e1, p=0.3,training=self.training)
        e1_2hop = self.get_2hop_idx(x, e1, batch)

        x_1 = self.conv1(x, e1)
        x_1 = self.bn1(x_1)
         
        x_1_1 = self.conv1_1(x, e1_2hop)
        x_1_1 = self.bn1(x_1_1)
        
        x_1 = x_1 + x_1_1
        # x_1 = torch.cat([x_1, x_1_1], dim=1)
        
        edge_index_2hop = self.get_2hop_idx(x, edge_index, batch)
        edge_index_2hop, _ = dropout_adj(edge_index_2hop, p=0.3,training=self.training)

        x_3 = self.conv3(x, edge_index)
        x_3 = self.bn3(x_3) 
        x_3_3 = self.conv3_3(x, edge_index_2hop)
        x_3_3 = self.bn3(x_3_3) 
        
        x1 = x_1 + x_3 + x_3_3

        h1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)
        # h1_2 = torch.cat([gmp(x_3, batch), gap(x_3, batch)], dim=1)

        e2, v2 = self.knn2(x, batch)#knn动态
        e2, _ = dropout_adj(e2, p=0.3,training=self.training)
        e2_2hop = self.get_2hop_idx(x, e2, batch)
        e2_2hop, _ = dropout_adj(e2_2hop, p=0.3,training=self.training)

        x_2 = self.conv2(x, e2)
        x_2 = self.bn2(x_2)
        x_2_2 = self.conv2_2(x, e2_2hop)
        x_2_2 = self.bn2(x_2_2)
        x_2 = x_2 + x_2_2
        # x_2 = torch.cat([x_2, x_2_2], dim=1)
        h2= x_2.view(x.shape[0]//116,116)
         
        fea = torch.cat([h1, h2], dim=-1)

        # if self.pool1_1:
        #     x_p, _, _, batch1, _, _ = self.pool1_1(x1, edge_index,edge_attr, batch=batch )        
        #     hp = torch.cat([gmp(x_p, batch1), gap(x_p, batch1)], dim=1)
        #     h1 = h1 + hp
        return  fea, x1
    
class NodeNorm(nn.Module):
    def __init__(self, nn_type="n", unbiased=False, eps=1e-5, power_root=2):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        x = x.view(x.shape[0]//116,116,-1)
        mean = torch.mean(x, dim=0, keepdim=True)
        std = (torch.var(x, unbiased=self.unbiased, dim=0, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        x = x.view(x.shape[0]*x.shape[1],-1)
        return x


