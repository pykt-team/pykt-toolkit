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

# class LSTMLayer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTMLayer, self).__init__()
#         # self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size, bias=False) for _ in range(num_layers)])
#         self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) if _ == 0 else nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

#     def forward(self, input_data):
        # batch_size, seq_length = input_data.size(0), input_data.size(1)
        # input_data = input_data.permute(1,0,2)
        # hidden_states = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        # cell_states = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        # # 前向传播
        # prev_hidden_states = []
        # for t in range(seq_length):
        #     for layer in range(self.num_layers):
        #         # 获取当前层的LSTMCell对象
        #         lstm_cell = self.lstm_cells[layer]
                
        #         # 获取当前时间步的输入数据
        #         if layer == 0:
        #             input_t = input_data[t]
        #         else:
        #             input_t = hidden_states[layer - 1]
                
        #         # 获取前一层的隐藏状态和细胞状态
        #         prev_hidden = hidden_states[layer - 1]
        #         prev_cell = cell_states[layer - 1]
        #         # prev_hidden = hidden_states[layer - 1] if layer > 0 else torch.zeros(batch_size, self.hidden_size).to(device)
        #         # prev_cell = cell_states[layer - 1] if layer > 0 else torch.zeros(batch_size, self.hidden_size).to(device)
                
        #         # LSTM单元前向传播
        #         hidden_states[layer], cell_states[layer] = lstm_cell(input_t, (prev_hidden, prev_cell))
        #     hidden_states_ = torch.stack(hidden_states, dim=0)
        #     prev_hidden_states.append(hidden_states_)
        
        # prev_hidden_states = torch.stack(prev_hidden_states, dim=0).permute(1,2,0,3) #[seqlen, num_layers, batch_size, hidden_size]
    
        # return prev_hidden_states
    

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        for _ in range(1, num_layers):
            self.layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs = []
        h, c = None, None
        for i in range(self.num_layers):
            x, (h, c) = self.layers[i](x)
            outputs.append(x)
        outputs = torch.stack(outputs, dim=0)
        # outputs,_ = self.lstm(x)
        return outputs

class GNN4KT(nn.Module):

    def __init__(self, n_question, n_pid, embed_l, hidden_size, dropout, num_layers=5, seq_len=200, 
    final_fc_dim=512, final_fc_dim2=256, emb_type="iekt", graph=None, emb_path="", pretrain_dim=768,mlp_layer_num=1): #
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
        self.num_layers = num_layers
        self.num_q = n_pid
        self.num_c = n_question

        self.que_emb = QueEmb(num_q=n_pid,num_c=n_question,emb_size=embed_l,emb_type=self.emb_type,model_name=self.model_name,device=device,
                    emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.qa_embed = nn.Embedding(2, embed_l)

        self.lstm_layer = LSTMLayer(input_size=embed_l, hidden_size=hidden_size, num_layers=num_layers)

        # self.graph = graph.to_dense()
        # # GCN for inter information
        # self.gnn_1 = GNNLayer(hidden_size, hidden_size) #一层gnn对应其中一个auto encoder
        # self.gnn_2 = GNNLayer(hidden_size, hidden_size)
        # self.gnn_3 = GNNLayer(hidden_size, hidden_size)
        # self.gnn_4 = GNNLayer(hidden_size, hidden_size)
        # self.gnn_5 = GNNLayer(hidden_size, hidden_size)

        # self.out = nn.Sequential(
        #     nn.Linear(embed_l*2,
        #             final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
        #     nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
        #     ), nn.Dropout(self.dropout),
        #     nn.Linear(final_fc_dim2, 1)
        # )
        self.out_question_all = MLP(mlp_layer_num, hidden_size+embed_l,self.num_q, self.dropout)
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
        # print(f"qa_embed_data:{qa_embed_data.shape}")

        # DNN Module
        # x_bar, tra1, tra2, tra3, z = self.ae(x) #获取auto-encoder得到emb

        bs, seqlen = pid_data.size(0), pid_data.size(1)

        #利用lstm得到每层的emb
        output = self.lstm_layer(qa_embed_data) #[num_layers, batch_size, seqlen, hidden_size]
        # print(f"output:{output.shape}")
        # output = torch.reshape(output, (self.num_layers, bs*seqlen, -1))
        sigma = 0.2

        # # # GCN Module
        # # # 根据当前数据构建小矩阵
        # idx = torch.reshape(pid_data,(1, bs*seqlen))[0]

        # # 根据索引 B 从矩阵 A 中提取值
        # # print(f"self.graph:{self.graph.shape}")
        # # print(f"pid:{torch.max(idx)}")
        # graph_ = self.graph[idx]
        # # print(f"graph_:{graph_.shape}")
        # sub_graph = graph_[:, idx]
        # # print(f"sub_graph:{sub_graph.shape}")


        # # 移动到数据加载
        
        # # # print(f"output:{output[0].shape}")
        # h = self.gnn_1(output[0], sub_graph)
        # # print(f"h:{h.shape}")
        # h = self.gnn_2((1-sigma)*h + sigma*output[1], sub_graph)
        # h = self.gnn_3((1-sigma)*h + sigma*output[2], sub_graph)
        # h = self.gnn_4((1-sigma)*h + sigma*output[3], sub_graph)
        # h = self.gnn_5((1-sigma)*h + sigma*output[4], sub_graph, active=False)
        # h = torch.reshape(h, (bs, seqlen, -1))
        # # print(f"h:{h.shape}")
        # input_combined = torch.cat((h[:,:-1,:], q_embed_data[:,1:,:]), -1)
        # h = output[4]

        h = output[4]
        input_combined = torch.cat((h[:,:-1,:], q_embed_data[:,1:,:]), -1)

        # input_combined = torch.cat((output[:,:-1,:], q_embed_data[:,1:,:]), -1)
        # y = self.out(input_combined)
        y = self.out_question_all(input_combined)
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
