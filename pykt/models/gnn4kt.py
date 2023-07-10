from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from .gnn4kt_util import GNNLayer
from .que_base_model import QueBaseModel,QueEmb
# from evaluation import eva
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device(1)

# auto-encoder
# class AE(nn.Module):

#     def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
#                  n_input, n_z):
#         super(AE, self).__init__()
#         self.enc_1 = Linear(n_input, n_enc_1)
#         self.enc_2 = Linear(n_enc_1, n_enc_2)
#         self.enc_3 = Linear(n_enc_2, n_enc_3)
#         self.z_layer = Linear(n_enc_3, n_z)

#         self.dec_1 = Linear(n_z, n_dec_1)
#         self.dec_2 = Linear(n_dec_1, n_dec_2)
#         self.dec_3 = Linear(n_dec_2, n_dec_3)
#         self.x_bar_layer = Linear(n_dec_3, n_input)

#     def forward(self, x):
#         enc_h1 = F.relu(self.enc_1(x))
#         enc_h2 = F.relu(self.enc_2(enc_h1))
#         enc_h3 = F.relu(self.enc_3(enc_h2))
#         z = self.z_layer(enc_h3)

#         dec_h1 = F.relu(self.dec_1(z))
#         dec_h2 = F.relu(self.dec_2(dec_h1))
#         dec_h3 = F.relu(self.dec_3(dec_h2))
#         x_bar = self.x_bar_layer(dec_h3)

#         return x_bar, enc_h1, enc_h2, enc_h3, z

class LSTMLayer(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layer):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layer, batch_first=True)

    def forward(self, input_emb):
        h, _ = self.lstm(input_emb)
        print(f"h:{h.shape}")
        print(f"hidden:{_[0].shape}")
        
        return h
    

class GNN4KT(nn.Module):

    def __init__(self, n_question, n_pid, embed_l, hidden_size, dropout, num_layers=5, seq_len=200, 
    final_fc_dim=512, final_fc_dim2=256, emb_type="iekt", graph=None, emb_path="", pretrain_dim=768):
        super(GNN4KT, self).__init__()

        # # autoencoder for intra information
        # self.ae = AE(
        #     n_enc_1=n_enc_1,
        #     n_enc_2=n_enc_2,
        #     n_enc_3=n_enc_3,
        #     n_dec_1=n_dec_1,
        #     n_dec_2=n_dec_2,
        #     n_dec_3=n_dec_3,
        #     n_input=n_input,
        #     n_z=n_z)
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        self.model_name = "gnn4kt"
        self.emb_type = emb_type
        self.dropout = dropout

        self.que_emb = QueEmb(num_q=n_pid,num_c=n_question,emb_size=embed_l,emb_type=self.emb_type,model_name=self.model_name,device=device,
                    emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.qa_embed = nn.Embedding(2, embed_l)

        self.lstm_layer = LSTMLayer(emb_size=embed_l, hidden_size=hidden_size, n_layer=num_layers)

        self.graph = graph
        # GCN for inter information
        self.gnn_1 = GNNLayer(hidden_size, hidden_size) #一层gnn对应其中一个auto encoder
        self.gnn_2 = GNNLayer(hidden_size, hidden_size)
        self.gnn_3 = GNNLayer(hidden_size, hidden_size)
        self.gnn_4 = GNNLayer(hidden_size, hidden_size)
        self.gnn_5 = GNNLayer(hidden_size, hidden_size)

        self.out = nn.Sequential(
            nn.Linear(embed_l*2,
                    final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.m = nn.Sigmoid()


    def forward(self, dcur):
        # input_data
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        # Batch First
        if pid_data.size(1) == 0:
            pid_data,q_data = q_data, pid_data
        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(pid_data, q_data, target)
        if q_data.size(1) == 0:
            q_embed_data = emb_q
        else:
            q_embed_data = emb_q + emb_c
        qa_embed_data = self.qa_embed(target)
        qa_embed_data = q_embed_data + qa_embed_data

        # DNN Module
        # x_bar, tra1, tra2, tra3, z = self.ae(x) #获取auto-encoder得到emb

        #利用lstm得到每层的emb
        output = self.lstm_layer(qa_embed_data)
        print(f"output:{output.shape}")
        print(f"output:{output[0].shape}")
        
        sigma = 0.5

        # GCN Module

        # 移动到数据加载
        h = self.gnn_1(output[0], self.graph)
        h = self.gnn_2((1-sigma)*h + sigma*output[1], self.graph)
        h = self.gnn_3((1-sigma)*h + sigma*output[2], self.graph)
        h = self.gnn_4((1-sigma)*h + sigma*output[3], self.graph)
        h = self.gnn_5((1-sigma)*h + sigma*output[4], self.graph, active=False)
        
        input_combined = torch.cat((h, q_embed_data[:,1:,:]), -1)
        y = self.out(input_combined)
        y = self.m(y)

        return y


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data) 

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703


    print(args)
    train_sdcn(dataset)
