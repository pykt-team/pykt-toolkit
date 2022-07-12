import random
def generate_postives(qss, rss, sms, num_q, probs={"mask": 0.2, "crop": 0.2, "permute": 0.2}):    
    totallen = qss.shape[1]
    finalqs, finalrs, finalsm = [], [], []
    # 也可以选一部分长度长的序列做对比学习
    for qs, rs, sm in zip(qss, rss, sms):
        sm = sm.bool()
        curqs = torch.masked_select(qs, sm).tolist()
        currs = torch.masked_select(rs, sm).tolist()
        cursm = torch.masked_select(sm, sm).tolist()
        curlen = len(curqs)
        if curlen < 30: # 
            continue
        # rand select
        flag = random.choice(list(probs.keys()))
        prob = probs[flag]
#         if int(curlen*prob) < 3: # 如果选中长度太短，crop和permute没啥意义
#             flag = "mask"
#             prob = probs[flag]
        # print("========")
        # print(f"flag: {flag}")
        if flag == "mask":
            idxs = [i for i in range(0, curlen)]
            select_num = max(int(curlen * prob), 1) # 至少mask 1个
            ids = random.sample(idxs, select_num)
            newqs = []
            for i in range(0, curlen):
                id = curqs[i]
                if i in ids:
                    id = num_q
                newqs.append(id)
            newrs, newsm = currs, cursm
        elif flag == "crop":
            # select subseq
            select_num = int(curlen*prob) if int(curlen*prob) > 2 else 2
            start = random.choice([i for i in range(0, curlen-select_num+1)])
            end = min(start + select_num, curlen)
            padlen = curlen - (end - start)
            newqs = curqs[start: end] + [0] * padlen
            newrs = currs[start: end] + [0] * padlen
            newsm = cursm[start: end] + [0] * padlen
        elif flag == "permute":
            # 
            select_num = int(curlen*prob) if int(curlen*prob) > 2 else 2
            start = random.choice([i for i in range(0, curlen-select_num+1)])
            
            end = min(start + select_num, curlen)
            subqs, subrs, subsms = curqs[start: end], currs[start: end], cursm[start: end]
            subinfos = []
            for subq, subr, subsm in zip(subqs, subrs, subsms):
                subinfos.append([subq, subr, subsm])
            random.shuffle(subinfos)
            newqs, newrs, newsm = [], [], []
            for info in subinfos:
                newqs.append(info[0])
                newrs.append(info[1])
                newsm.append(info[2])
            newqs = curqs[0: start] + newqs + curqs[end: ]
            newrs = currs[0: start] + newrs + currs[end: ]
            newsm = cursm[0: start] + newsm + cursm[end: ]
        padlen = totallen - curlen
        newqs, newrs, newsm = newqs + [0] * padlen, newrs + [0] * padlen, newsm + [0] * padlen
        
        finalqs.append(newqs)
        finalrs.append(newrs)
        finalsm.append(newsm)
    
    finalqs, finalrs, finalsm = torch.LongTensor(finalqs), torch.LongTensor(finalrs), torch.BoolTensor(finalsm)
    return finalqs, finalrs, finalsm

from torch import nn
from torch.nn.functional import normalize
class Network(nn.Module):
    def __init__(self, net, net_type, net_out_dim, feature_dim, class_num, drop_rate=0.1, instance_temp=0.5, cluster_temp=1):
        super(Network, self).__init__()
        self.net = net
        self.net_type = net_type
        # self.dropout = nn.Dropout(drop_rate)
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(net_out_dim, net_out_dim),
            nn.ReLU(),
            nn.Linear(net_out_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(net_out_dim, net_out_dim),
            nn.ReLU(),
            nn.Linear(net_out_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.instance_loss = InstanceLoss(instance_temp)
        self.cluster_loss = ClusterLoss(class_num, cluster_temp)

    def forward(self, x_i, x_j, sm_i, sm_j):
        # x_i : bz * channel * width * higth
        # 输入是一个bz的样本经过不同数据增强方式得到的两种增强结果
        # 新: [id1, id2.....], [id1, id2.....]
        if self.net_type == "lstm":
            h_i, _ = self.net(x_i)
            h_j, _ = self.net(x_j)
            h_i, h_j = h_i[:,-1,:], h_j[:,-1,:]
        elif self.net_type == "transformer":
            h_i = self.net(x_i.transpose(0,1), sm_i).transpose(0,1)
            h_j = self.net(x_j.transpose(0,1), sm_j).transpose(0,1)
            # print(f"h_i: {h_i.shape}")
            h_i, h_j = h_i[:,0,:], h_j[:,0,:]
#         print(f"h_i: {h_i}")
#         print(f"h_j: {h_j}")

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        loss_instance = self.instance_loss(z_i, z_j)
        loss_cluster = self.cluster_loss(c_i, c_j)
        loss = loss_instance + loss_cluster

        return loss

    def forward_cluster(self, x):
        h = self.net(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

class WWWNetwork(nn.Module):
    # change to real www!
    def __init__(self, qnet, xnet, net_type, net_out_dim, feature_dim, class_num, drop_rate=0.1, instance_temp=0.5, cluster_temp=1):
        super(WWWNetwork, self).__init__()
        self.qnet = qnet
        self.xnet = xnet
        self.net_type = net_type
        self.feature_dim = feature_dim
        # self.pooling = 
        self.instance_projector = nn.Sequential(
            nn.Linear(net_out_dim, net_out_dim),
            nn.ReLU(),
            nn.Linear(net_out_dim, self.feature_dim),
        )
        self.qloss = InstanceLoss(instance_temp)
        self.xloss = InstanceLoss(instance_temp)

    def forward(self, q_i, q_j, x_i, x_j, sm_i, sm_j):
        # x_i : bz * channel * width * higth
        # 输入是一个bz的样本经过不同数据增强方式得到的两种增强结果
        # 新: [id1, id2.....], [id1, id2.....]
        def geth(x_i, x_j, sm_i, sm_j, net):
            if self.net_type == "lstm":
                h_i, _ = net(x_i)
                h_j, _ = net(x_j)
                h_i, h_j = h_i[:,-1,:], h_j[:,-1,:]
            elif self.net_type == "transformer":
                h_i = net(x_i.transpose(0,1), sm_i).transpose(0,1)
                h_j = net(x_j.transpose(0,1), sm_j).transpose(0,1)
                # print(f"h_i: {h_i.shape}")
                h_i, h_j = h_i[:,0,:], h_j[:,0,:]
            z_i = normalize(self.instance_projector(h_i), dim=1)
            z_j = normalize(self.instance_projector(h_j), dim=1)
            return z_i, z_j
        zq_i, zq_j = geth(q_i, q_j, sm_i, sm_j, self.qnet)
        zx_i, zx_j = geth(x_i, x_j, sm_i, sm_j, self.xnet)
        qloss = self.qloss(zq_i, zq_j)
        xloss = self.xloss(zx_i, zx_j)
        loss = qloss+xloss

        return loss

import torch
import torch.nn as nn
import math
class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        curbz = z_j.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        # print(f"z_i: {z_i.shape}, z_j: {z_j.shape}, z: {z.shape}, z.T {z.T.shape}")
        sim = torch.matmul(z, z.T) / self.temperature        
        N = 2 * curbz
        mask = self.mask_correlated_samples(curbz)
        sim_i_j = torch.diag(sim, curbz)
        sim_j_i = torch.diag(sim, -curbz)
        
        try:
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[mask].reshape(N, -1)
        except Exception as e:
            print(f"z_i: {z_i.shape}, z_j: {z_j.shape}")
            print(f"sim_i_j: {sim_i_j.shape}, sim_j_i: {sim_j_i.shape}")
            print(f"mask: {mask.shape}")
            print(f"sim: {sim.shape}")
            print(f"sim[mask]: {sim[mask]}")
            print(f"mask: {mask}")
            print(e)
            assert False

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # print(f"positive_samples: {positive_samples.shape}, negative_samples: {negative_samples.shape}, logits: {logits.shape}, labels: {labels.shape}, labels: {labels.tolist()}")
        # print(logits)
        # assert False
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss