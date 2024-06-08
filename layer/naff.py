# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, adj, args, user_feat=None, item_feat=None):

        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.decay = args.decay
        self.layers = args.layers
        self.adj = adj

        # self.sim = sim
        adj_n = self.adj.to_dense()
        adj_n[adj_n > 0] = 1
        self.adj_n = torch.sum(adj_n, dim=1, keepdim=True)
        self.adj_n = torch.log(self.adj_n)
        adj_mean = torch.mean(self.adj_n, dim=0)
        self.adj_n = self.adj_n / adj_mean

        # sim_n = self.sim.to_dense()
        # sim_n[sim_n > 0] = 1
        # self.sim_n = torch.sum(sim_n, dim=1, keepdim=True)
        # sim_mean = torch.mean(self.sim_n, dim=0)
        # self.sim_n = self.sim_n/sim_mean

        user_emb_weight = self._weight_init(user_feat, n_users, args.emb_size)
        item_emb_weight = self._weight_init(item_feat, n_items, args.emb_size)

        self.user_embeddings = nn.Embedding(self.n_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_size, _weight=item_emb_weight)

        # self.lightgcn_amazonbook_embedding_file_user = '/home/yjh/Model-2/GNNEC/dataset/amazon-book/pretrain-embeddings/GNNEC/n_layers=2/user_embeddings.npy'
        # self.lightgcn_amazonbook_embedding_file_item = '/home/yjh/Model-2/GNNEC/dataset/amazon-book/pretrain-embeddings/GNNEC/n_layers=2/item_embeddings.npy'
        # user_pre = np.load(self.lightgcn_amazonbook_embedding_file_user)
        # user_pre = torch.tensor(user_pre).to(device)
        #
        # item_pre = np.load(self.lightgcn_amazonbook_embedding_file_item)
        # item_pre = torch.tensor(item_pre).to(device)
        # self.pre_emb = torch.cat([user_pre, item_pre], dim=0)
        # self.pre_emb = F.normalize(self.pre_emb, p=2, dim=1)


    def _weight_init(self, feat, rows, cols):

        if feat is None:
            free_emb = nn.init.normal_(torch.empty(rows, cols), std=0.01)
            return free_emb
        else:
            free_emb = nn.init.normal_(torch.empty(rows, cols - feat.shape[-1]), std=0.01)
            feat_emb = torch.tensor(feat) * 0.01
            return torch.cat([free_emb, feat_emb], dim=1)

    def forward(self, adj):

        x = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb0 = []
        all_emb = []

        for i in range(self.layers):
            x_norm = F.normalize(x, p=2, dim=1)
            x = torch.sparse.mm(adj, x)
            x1_norm = F.normalize(x, p=2, dim=1)
            x2 = torch.sum(x_norm*x1_norm, dim=1, keepdim=True) #计算相似度
            x2 = torch.clamp(x2, min=0)
            x2 = torch.exp(1/(3*x2 + self.adj_n*(1+i))) #计算融合权重, self.adj是节点活跃度
            x0 = x*x2
            all_emb += [x0]
            all_emb0 += [x]

        return all_emb, x

    def fusion(self, embeddings):

        # differ from lr-gccf
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings

    def split_emb(self, embeddings):

        user_emb, item_emb = torch.split(embeddings, [self.n_users, self.n_items])
        return user_emb, item_emb

    def get_embedding(self, user_emb, item_emb, users, pos_items, neg_items):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        return u_emb, pos_emb, neg_emb

    def propagate(self):
        all_emb, all_emb0 = self.forward(self.adj)
        f_emb = self.fusion(all_emb)
        # f_emb0 = self.fusion(all_emb0)
        user_emb, item_emb = self.split_emb(f_emb)
        user_emb0, item_emb0 = self.split_emb(all_emb0)

        return user_emb, item_emb, user_emb0, item_emb0

    def creat_infoloss(self, a_embeddings, b_embeddings,
                       ssl_temp, ssl_reg):
        # argument settings

        user_emb1 = torch.nn.functional.normalize(a_embeddings, p=2, dim=1)
        user_emb2 = torch.nn.functional.normalize(b_embeddings, p=2, dim=1)
        # user
        user_pos_score = torch.multiply(user_emb1, user_emb2).sum(dim=1)
        user_ttl_score = torch.matmul(user_emb1, user_emb2.t())
        user_pos_score = torch.exp(user_pos_score / ssl_temp)
        user_ttl_score = torch.exp(user_ttl_score / ssl_temp).sum(dim=1)
        user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()
        ssl_loss = ssl_reg * user_ssl_loss

        return ssl_loss

    def KLDiverge(self, tpreds, spreds, distillTemp, w):
        tpreds = (tpreds / distillTemp).sigmoid()
        spreds = (spreds / distillTemp).sigmoid()
        return w*(-(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean())

    def bpr_loss(self, user_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):

        pos_score = torch.sum(user_emb * pos_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_emb, dim=1)
        item_score = torch.sum(pos_emb * neg_emb, dim=1)

        pos_score0 = torch.sum(user_emb0 * pos_emb0, dim=1)
        neg_score0 = torch.sum(user_emb0 * neg_emb0, dim=1)
        item_score0 = torch.sum(pos_emb0 * neg_emb0, dim=1)
        nn = pos_score - neg_score
        mm = pos_score0 - neg_score0
        # dist = self.KLDiverge(nn, mm, 1, 1)
        # ssl_user = self.creat_infoloss(user_emb, user_emb0, 0.05, 0.00001)
        # ssl_item = self.creat_infoloss(pos_emb, pos_emb0, 0.05, 0.00001)
        dist = self.KLDiverge(nn, mm, 1, 1)#预测级对比学习
        ssl_user = self.creat_infoloss(user_emb, user_emb0, 0.05, 0.00001)#用户表示级对比学习
        ssl_item = self.creat_infoloss(pos_emb, pos_emb0, 0.05, 0.00001)#物品表示级对比学习

        # NAFF
        #mf_loss = torch.mean(F.softplus((neg_score - pos_score))) # 不使用对比学习NAFF

        # NAFFCL
        mf_loss = torch.mean(F.softplus((neg_score + item_score - pos_score))) + ssl_item + ssl_user  + dist


        reg_loss = (1/2) * (user_emb.norm(2).pow(2) +
                            pos_emb.norm(2).pow(2) +
                            neg_emb.norm(2).pow(2)) / user_emb.shape[0] * self.decay

        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss