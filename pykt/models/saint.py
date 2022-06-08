import torch 
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear

device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAINT(nn.Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        print(f"num_q: {num_q}, num_c: {num_c}")
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "saint"
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type

        self.embd_pos = nn.Embedding(seq_len, embedding_dim = emb_size) 
        # self.embd_pos = Parameter(torch.Tensor(seq_len-1, emb_size))
        # kaiming_normal_(self.embd_pos)

        if emb_type.startswith("qid"):
            self.encoder = get_clones(Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout), self.num_en)
        
        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)
    
    def forward(self, in_ex, in_cat, in_res, qtest=False):
        emb_type = self.emb_type        

        if self.num_q > 0:
            in_pos = pos_encode(in_ex.shape[1])
        else:
            in_pos = pos_encode(in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if emb_type == "qid": # same to qid in saint
                in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(in_res.shape[0], 1).to(device)
        in_res = torch.cat((start_token, in_res), dim=-1)
        r = in_res
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](in_res, in_pos, en_out=in_ex, first_block=first_block)
        
        ## Output layer

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)
        if not qtest:
            return res
        else:
            return res, in_res


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim = dim_model)                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim = dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim = dim_model, num_heads = heads_en, dropout = dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex
        
        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape)
        
        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en(out, out, out,
                                attn_mask=ut_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = out + skip_out                                    # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2(out)                           # Layer norm 
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_res    = nn.Embedding(total_res+1, embedding_dim = dim_model)                  #response embedding, include a start token
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)


    def forward(self, in_res, in_pos, en_out,first_block=True):

         ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            in_in = self.embd_res(in_res)

            #combining the embedings
            out = in_in + in_pos                         # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape)
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, 
                                     attn_mask=ut_mask(seq_len=n)) # attention mask upper triangular
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                    attn_mask=ut_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out
