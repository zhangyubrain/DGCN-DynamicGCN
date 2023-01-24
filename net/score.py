# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:25:52 2020

@author: 99488
"""
import torch
from torch.nn import Parameter
from torch_geometric.utils import softmax

from utils.inits import uniform
#modified from topk function in torch_geometric
class Get_score(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers
    if min_score :math:`\tilde{\alpha}` is None:
        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}
            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}
    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:
        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})
            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}
            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},
    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh):
        super(Get_score, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        #####  zero mean for each instance #########3
        score = score.view(batch.max() + 1, -1)
        score = score - score.mean(1,keepdim=True) #
        score = score.view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        x = x * score.view(-1, 1)

        return x, score.view(batch.max()+1,-1)


    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)
