import torch

from .akt import AKT
from .opera_utils import OperaQuestionEmbeddingMixin


class AKT_Enhance_Pro_qid(OperaQuestionEmbeddingMixin, AKT):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
                 kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False,
                 l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, data_config=None):
        if data_config is None:
            raise ValueError("akt_enhance_pro_qid requires data_config with pro_emb_path.")

        super().__init__(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
        )
        self.model_name = "akt_enhance_pro_qid"
        self.model_type = "akt"
        self.setup_question_embeddings(data_config, d_model, n_pid, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def base_emb(self, q_data, target):
        q_embed_data = self.get_question_embedding(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(pid_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.get_question_embedding(pid_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))
        if not qtest:
            return preds, c_reg_loss
        return preds, c_reg_loss, concat_q
