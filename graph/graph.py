# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Graph(object):

    def __init__(self, n_users, n_items, train_U2I):

        self.n_users = n_users
        self.n_items = n_items
        self.train_U2I = train_U2I

    def to_edge(self):

        train_U, train_I = [], []

        for u, items in self.train_U2I.items():
            train_U.extend([u] * len(items))
            train_I.extend(items)

        train_U = np.array(train_U)
        train_I = np.array(train_I)

        row = np.concatenate([train_U, train_I + self.n_users])
        col = np.concatenate([train_I + self.n_users, train_U])

        edge_weight = np.ones_like(row).tolist()
        edge_index = np.stack([row, col]).tolist()

        return edge_index, edge_weight


class LaplaceGraph(Graph):

    def __init__(self, n_users, n_items, train_U2I):
        Graph.__init__(self, n_users, n_items, train_U2I)

    def generate(self):
        edge_index, edge_weight = self.to_edge()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        return self.mat(edge_index, edge_weight)
        # # 生成稀疏矩阵
        # sparse_mat = self.mat(edge_index, edge_weight)
        # # # 稀疏矩阵转为稠密矩阵，并计算邻居重要性
        # dense_mat = sparse_mat.to_dense()
        # dense_mat_norm = F.normalize(dense_mat, p=2, dim=1)
        # dense = torch.matmul(dense_mat_norm, dense_mat_norm.T)
        # dense = (dense+1)/2
        # # dense = torch.randn(self.num_nodes, self.num_nodes)
        # dense[dense < 0.95] = 0
        # dense[dense >= 0.95] = 1
        #
        # dense = dense + torch.eye(len(dense_mat)) + dense_mat
        # # 归一化，并返回稀疏矩阵
        # D = torch.sum(dense, dim=1).float()
        # D[D == 0.] = 1.
        # D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        # dense = dense / D_sqrt
        # dense = dense / D_sqrt.t()
        # index = dense.nonzero()
        # data = dense[dense > 1e-9]
        # assert len(index) == len(data)
        # edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        # edge_index, edge_weight = self.norm(edge_index, edge_weight)
        # return self.mat(edge_index, edge_weight), torch.sparse.FloatTensor(index.t(), data, torch.Size([self.num_nodes, self.num_nodes]))


    def add_self_loop(self, edge_index, edge_weight):
        """ add self-loop """

        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        """ D^{-1/2} * A * D^{-1/2}"""

        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # edge_weight = edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.n_users + self.n_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))
