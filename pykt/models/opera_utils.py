import json
import os

import numpy as np
import torch
from torch.nn import Embedding, Linear


OPERA_ENHANCE_PRO_MODELS = {
    "dkt_enhance_pro",
    "dkvmn_enhance_pro",
    "sakt_enhance_pro",
    "akt_enhance_pro_qid",
    "simplekt_enhance_pro_qid",
}

SUPPORTED_OPERA_DATASETS = {
    "jiuzhang_grade3_en": "3",
    "jiuzhang_grade45_cn": "45",
    "jiuzhang_grade7_cn": "7",
}


def _resolve_existing_path(path):
    if path and os.path.exists(path):
        return path
    if path and path.startswith("../data/"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        project_data_path = os.path.join(project_root, "data", path[len("../data/"):])
        if os.path.exists(project_data_path):
            return project_data_path
    return path


def validate_opera_dataset(model_name, data_config):
    dataset_name = data_config.get("dataset_name")
    if dataset_name is None and data_config.get("dpath"):
        dataset_name = os.path.basename(os.path.normpath(data_config["dpath"]))
    if dataset_name not in SUPPORTED_OPERA_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_OPERA_DATASETS))
        raise ValueError(
            f"{model_name} requires question text embeddings and only supports: {supported}. "
            f"Got dataset_name={dataset_name!r}."
        )

    pro_emb_path = _resolve_existing_path(data_config.get("pro_emb_path"))
    if not pro_emb_path or not os.path.exists(pro_emb_path):
        raise ValueError(f"{model_name} requires an existing pro_emb_path. Got {pro_emb_path!r}.")


class OperaQuestionEmbeddingMixin:
    def setup_question_embeddings(self, data_config, emb_size, num_q, device):
        validate_opera_dataset(self.model_name, data_config)
        self.semantic_emb_path = _resolve_existing_path(data_config["pro_emb_path"])
        self.no_q_path = _resolve_existing_path(data_config.get("no_q_path", ""))
        self.no_semantic_questions = set()

        if self.no_q_path and os.path.exists(self.no_q_path):
            with open(self.no_q_path, "r", encoding="utf8") as fin:
                self.no_semantic_questions = set(json.load(fin))

        semantic_embeddings = np.load(self.semantic_emb_path)
        if semantic_embeddings.shape[0] < num_q:
            raise ValueError(
                f"{self.model_name} semantic embeddings at {self.semantic_emb_path} have "
                f"{semantic_embeddings.shape[0]} rows, but num_q={num_q}."
            )

        semantic_embeddings = torch.FloatTensor(semantic_embeddings)
        self.semantic_emb = Embedding.from_pretrained(semantic_embeddings, freeze=False).to(device)
        self.random_question_emb = Embedding(num_q, emb_size).to(device)
        if semantic_embeddings.shape[1] != emb_size:
            self.semantic_proj = Linear(semantic_embeddings.shape[1], emb_size).to(device)
        else:
            self.semantic_proj = None

    def get_question_embedding(self, q_ids):
        q_ids = q_ids.to(self.semantic_emb.weight.device)
        question_emb = self.semantic_emb(q_ids)
        if self.semantic_proj is not None:
            question_emb = self.semantic_proj(question_emb)

        if self.no_semantic_questions:
            no_semantic_mask = torch.zeros_like(q_ids, dtype=torch.bool)
            for no_q_id in self.no_semantic_questions:
                no_semantic_mask |= q_ids == no_q_id
            if no_semantic_mask.any():
                question_emb[no_semantic_mask] = self.random_question_emb(q_ids[no_semantic_mask])

        return question_emb
