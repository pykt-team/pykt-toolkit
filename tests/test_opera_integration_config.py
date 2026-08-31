def test_opera_enhance_pro_models_use_question_level_dataloader():
    from pykt.config import que_type_models

    assert {
        "dkt_enhance_pro",
        "dkvmn_enhance_pro",
        "sakt_enhance_pro",
        "akt_enhance_pro_qid",
        "simplekt_enhance_pro_qid",
    }.issubset(set(que_type_models))
