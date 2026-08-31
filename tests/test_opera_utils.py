import importlib.util
from pathlib import Path

import pytest


def load_opera_utils():
    path = Path(__file__).resolve().parents[1] / "pykt" / "models" / "opera_utils.py"
    spec = importlib.util.spec_from_file_location("opera_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opera_dataset_validation_rejects_unsupported_dataset():
    validate_opera_dataset = load_opera_utils().validate_opera_dataset

    with pytest.raises(ValueError, match="only supports"):
        validate_opera_dataset(
            "dkt_enhance_pro",
            {
                "dataset_name": "assist2009",
                "pro_emb_path": "missing.npy",
            },
        )


def test_opera_dataset_validation_requires_embedding_path_for_supported_dataset(tmp_path):
    validate_opera_dataset = load_opera_utils().validate_opera_dataset

    missing_path = tmp_path / "question_embeddings_overall.npy"

    with pytest.raises(ValueError, match="pro_emb_path"):
        validate_opera_dataset(
            "dkt_enhance_pro",
            {
                "dataset_name": "jiuzhang_grade3_en",
                "pro_emb_path": str(missing_path),
            },
        )


def test_opera_dataset_validation_accepts_supported_dataset_with_embedding(tmp_path):
    validate_opera_dataset = load_opera_utils().validate_opera_dataset

    emb_path = tmp_path / "question_embeddings_overall.npy"
    emb_path.write_bytes(b"placeholder")

    validate_opera_dataset(
        "dkt_enhance_pro",
        {
            "dataset_name": "jiuzhang_grade3_en",
            "pro_emb_path": str(emb_path),
        },
    )
