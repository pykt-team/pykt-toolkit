import torch

from .opera_utils import OperaQuestionEmbeddingMixin
from .simplekt import simpleKT


class simpleKT_enhance_pro_qid(OperaQuestionEmbeddingMixin, simpleKT):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
                 loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2,
                 nheads=4, seq_len=200, kq_same=1, final_fc_dim=512,
                 final_fc_dim2=256, num_attn_heads=8, separate_qa=False,
                 l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, data_config=None):
        if data_config is None:
            raise ValueError("simplekt_enhance_pro_qid requires data_config with pro_emb_path.")

        super().__init__(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            loss1=loss1,
            loss2=loss2,
            loss3=loss3,
            start=start,
            num_layers=num_layers,
            nheads=nheads,
            seq_len=seq_len,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
        )
        self.model_name = "simplekt_enhance_pro_qid"
        self.model_type = "simplekt"
        self.setup_question_embeddings(data_config, d_model, n_pid, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def base_emb(self, q_data, target):
        q_embed_data = self.get_question_embedding(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(pid_data, target)

        if self.n_pid > 0 and self.emb_type.find("norasch") == -1:
            q_embed_diff_data = self.get_question_embedding(pid_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            if self.emb_type.find("aktrasch") != -1:
                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        y2, y3 = 0, 0
        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))

        if train:
            return preds, y2, y3
        if qtest:
            return preds, concat_q
        return preds
