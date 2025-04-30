import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCNConv(nn.Module):  
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)).to(device))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)).to(device))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p).to(device)

    def forward(self, x, adj):
        x = self.dropout(x.to(device))
        x = torch.matmul(x, self.w.to(device))
        x = torch.sparse.mm(adj.float().to(device), x)
        x = x + self.b.to(device)
        return x.to(device)


class MLP_Predictor(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True).to(device),
            nn.BatchNorm1d(hidden_size).to(device),
            nn.PReLU().to(device),
            nn.Linear(hidden_size, output_size, bias=True).to(device)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x.to(device))


def loss_fn(x, y):  
    x = F.normalize(x.to(device), dim=-1, p=2)
    y = F.normalize(y.to(device), dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def get_kc_embedding(last_kc, kc_emb, padding_idx=-1):
    batch_size, seq_len, max_concepts = last_kc.shape
    dim = kc_emb.size(1)
    device = last_kc.device
    
    mask = (last_kc != padding_idx)  # [batch_size, seq_len, max_concepts]
    
    last_kc = last_kc.clamp(min=0)
    
    flat_kc = last_kc.view(-1)  # [batch_size * seq_len * max_concepts]
    flat_emb = F.embedding(flat_kc, kc_emb)  # [batch_size * seq_len * max_concepts, dim]

    emb = flat_emb.view(batch_size, seq_len, max_concepts, dim)  # [batch_size, seq_len, max_concepts, dim]
    
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, dim)  # [batch_size, seq_len, max_concepts, dim]
    
    masked_emb = emb * mask.float()
    
    concept_counts = mask.sum(dim=2, keepdim=True)  # [batch_size, seq_len, 1, dim]
    
    concept_counts = concept_counts.clamp(min=1.0)
    
    pooled_emb = masked_emb.sum(dim=2) / concept_counts.squeeze(2)  # [batch_size, seq_len, dim]

    pooled_emb = pooled_emb.squeeze(0)  # [seq_len, dim]

    return pooled_emb  

class BGRL(nn.Module):  
    def __init__(self, d, p):
        super(BGRL, self).__init__()

        self.online_encoder = GCNConv(d, d, p).to(device)  

        self.decoder = GCNConv(d, d, p).to(device)

        self.predictor = MLP_Predictor(d, d, d).to(device)

        self.target_encoder = copy.deepcopy(self.online_encoder).to(device)

        self.fc1 = nn.Linear(d, d).to(device)
        self.fc2 = nn.Linear(d, d).to(device)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)


    def forward(self, x, adj, perb=None):
        if perb is None:  
            return (x + self.online_encoder(x, adj)).to(device), torch.tensor(0.0).to(device)

        x1, adj1 = x.to(device), copy.deepcopy(adj).to(device)
        x2, adj2 = (x + perb).to(device), copy.deepcopy(adj).to(device)

        embed = (x2 + self.online_encoder(x2, adj2)).to(device)   

        online_x = self.online_encoder(x1, adj1).to(device)
        online_y = self.online_encoder(x2, adj2).to(device)

        with torch.no_grad():
            target_y = self.target_encoder(x1, adj1).detach().to(device)
            target_x = self.target_encoder(x2, adj2).detach().to(device)

        online_x = self.predictor(online_x).to(device)
        online_y = self.predictor(online_y).to(device)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean().to(device)

        return embed, loss



class SFM_CL(nn.Module):
    def __init__(self, skill_max, positive_matrix, pro_max, d, p, dataset_name):
        super(SFM_CL, self).__init__()

        self.d_model = d
        self.pro_max = pro_max

        self.gcl = BGRL(d, p).to(device)

        self.gcn = GCNConv(d, d, p).to(device)

        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)).to(device))  
        self.ans_embed = nn.Embedding(2, d).to(device)  
        self.change = nn.Linear(1024, d).to(device)

        # the file_path can be found in :
        # https://drive.google.com/drive/folders/1cUqLbBRlj_PPIIhySghyaIjlasIDGIwF?usp=drive_link

        file_path = f'../data/{dataset_name}/question_concept_map.npy'
        concept_map = torch.tensor(np.load(file_path)).unsqueeze(0) # [1, pro_max, max_concept]

        file_path = f'../KC_Text_change/data/kc_embeddings_{dataset_name}_bge.npy'
        concept_embedding = torch.tensor(np.load(file_path))
        pro_embedding_kc = get_kc_embedding(concept_map, concept_embedding).to(device)  # [content_num, 1024]
        pro_embedding_kc = self.change(pro_embedding_kc).to(device)  # [pro_max, d]
        self.pro_embed = nn.Parameter(pro_embedding_kc * self.pro_embed).to(device)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                

    def forward(self, last_pro, last_ans, next_pro, matrix, perb=None):

        pro_embed, contrast_loss = self.gcl(self.pro_embed, matrix, perb)
        pro_embed = pro_embed.to(device)
        contrast_loss = 0.1 * contrast_loss.to(device)  

        last_pro_embed = F.embedding(last_pro, pro_embed).to(device)  # [80,199,128]
        next_pro_embed = F.embedding(next_pro, pro_embed).to(device)  # [80,199,128]

        ans_embed = self.ans_embed(last_ans).to(device)

        X = last_pro_embed.to(device)
        
        return X, next_pro_embed, contrast_loss

