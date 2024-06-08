# -*- coding: utf-8 -*-

root = '/home/yjh/Model-1/gnn-basemodel/NAFF/'

import sys
sys.path.append(root)
import os
import random
import torch
import numpy as np
import argparse
import torch.optim as optim
from graph.graph import LaplaceGraph
from codes.dataset import FeaturesData, BPRTrainLoader, UserItemData
from codes.performance import evaluate
from layer.naff import LightGCN
from session.run import Session
from torch.utils.data import DataLoader
# seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
mydataset = 'amazon-book'

def parse_args():
    parser = argparse.ArgumentParser(description="gcn")
    parser.add_argument('--dataset_name', default=mydataset, type=str)
    parser.add_argument('--data_path', default=root + '/data/', type=str)
    parser.add_argument('--emb_size', default=32, type=int)
    parser.add_argument('--num_epoch', default=300, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=0.001, type=float)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--topks', default='[10,20]', type=str)
    parser.add_argument('--log', default=root + '/log/{}_naff.txt'.format(mydataset), type=str)
    parser.add_argument('--parameters_path', default= root + '/save_parmeters/', type=str)
    parser.add_argument('--cores', default=1, type=int)
    parser.add_argument('--test_flag', default=0, type=int)
    return parser.parse_args()


def get_dataloader(train_set, train_U2I, n_items, batch_size, cores):
    gcn_dataloader = BPRTrainLoader(train_set, train_U2I, n_items)
    gcn_dataloader = DataLoader(gcn_dataloader, batch_size, num_workers=cores, shuffle=True)

    return gcn_dataloader

def test(model, n_users, n_items):

    user_emb, item_emb, user_emb0, item_emb0 = model.propagate()
    return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()


if __name__ == '__main__':
    args = parse_args()

    # load movielens-1m
    # data = FeaturesData(args.data_path, args.dataset_name)
    # train_set, train_U2I, test_U2I, n_users, n_items, user_feat = data.load()
    print(mydataset)
    # load gowalla
    data = UserItemData(args.data_path, args.dataset_name)
    train_set, train_U2I, test_U2I, n_users, n_items = data.load()
    print('user_nums:', n_users, 'item_nums:', n_items)
    loader = get_dataloader(train_set, train_U2I, n_items, args.batch_size, args.cores)
    g = LaplaceGraph(n_users, n_items, train_U2I)
    adj = g.generate().cuda()
    # adj = adj.cuda()
    # sim = sim.cuda()

    gcn = LightGCN(n_users, n_items, adj, args)
    gcn = gcn.cuda()

    optimizer = optim.Adam(gcn.parameters(), lr=args.lr)

    if args.test_flag:
        gcn.load_state_dict(torch.load(root + '/save_model/test.pt'))
        gcn.eval()

        user_emb, item_emb = test(gcn, n_users, n_items)
        groupid = [0,25,30,40,50,60,70]
        # groupid = [0,35,40,50,60,70]
        for i in range(len(groupid)-1):
            user_file = args.data_path + '/' + args.dataset_name + '/longtail_user/%d~%d.npy' % (groupid[i], groupid[i+1])
            test_user = np.load(user_file)
            longtail_test_U2I = {}
            for u in test_user:
                longtail_test_U2I[u] = test_U2I[u]

            perf_info = evaluate(user_emb, item_emb, n_users, n_items, train_U2I, longtail_test_U2I, args)
            print("recall@10:[{:.6f}], ndcg@10:[{:.6f}], recall@20:[{:.6f}], ndcg@20:[{:.6f}]".format(*perf_info))
    else:
        sess = Session(gcn)

        f = open(args.log, 'w+')
        for epoch in range(args.num_epoch):

            loss = sess.train(loader, optimizer, args)
            print("epoch:{:d}, loss:[{:.6f}] = mf:[{:.6f}] + reg:[{:.6f}]".format(epoch+1, *loss))
            print("epoch:{:d}, loss:[{:.6f}] = mf:[{:.6f}] + reg:[{:.6f}]".format(epoch+1, *loss), file=f)

            gcn.eval()
            with torch.no_grad():
                user_emb, item_emb = test(gcn, n_users, n_items)
                perf_info = evaluate(user_emb,
                                    item_emb,
                                    n_users,
                                    n_items,
                                    train_U2I,
                                    test_U2I,
                                    args)

                print("recall@10:[{:.6f}], ndcg@10:[{:.6f}], recall@20:[{:.6f}], ndcg@20:[{:.6f}]".format(*perf_info), file=f)
                print("recall@10:[{:.6f}], ndcg@10:[{:.6f}], recall@20:[{:.6f}], ndcg@20:[{:.6f}]".format(*perf_info))

        f.close()
        torch.save(gcn.state_dict(), root + '/save_model/test.pt')
        # save embedding
        # user_emb, item_emb = test(gcn, n_users, n_items)
        # np.save(root + '/save_model/' + "user_embedding.npy",user_emb) # 大功告成
        # torch.save((user_emb, item_emb), f= root + '/save_model/embedding' + '.pth')
