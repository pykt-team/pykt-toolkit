#!/bin/bash
# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_bayesian_v2_deepbkt_tiaocan_assist2009_42_0_0.25_512_512_512_8_2_0.0001_0.5_0.05_0.2_1_1_a0b428ba-4b7b-477b-875c-7ee00b0461b7/ ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_bayesian_v2_deepbkt_tiaocan_assist2009_42_0_0.25_512_512_512_8_2_0.0001_0.5_0.05_0.2_1_1_a0b428ba-4b7b-477b-875c-7ee00b0461b7 --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_bayesian_deepbkt_tiaocan_assist2009_224_0_0.2_256_512_256_4_3_0.0001_0.5_0.1_0.1_1_1_3a8a81c9-6348-445b-97bb-f350347e3aec/ ./saved_model/
# CUDA_VISIBLE_DEVICES=0 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_bayesian_deepbkt_tiaocan_assist2009_224_0_0.2_256_512_256_4_3_0.0001_0.5_0.1_0.1_1_1_3a8a81c9-6348-445b-97bb-f350347e3aec --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_1_0.15_256_512_256_4_4_0.0001_0.5_0.01_0.05_1_1_dd7a0e69-319b-436c-98a9-5f8541bc5cef/ ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_1_0.15_256_512_256_4_4_0.0001_0.5_0.01_0.05_1_1_dd7a0e69-319b-436c-98a9-5f8541bc5cef --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_3407_2_0.2_256_512_256_4_3_0.0001_0.1_0.025_0.1_1_1_0fafbf1a-41c6-4cee-a0ea-44fec6fccc9b/ ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_3407_2_0.2_256_512_256_4_3_0.0001_0.1_0.025_0.1_1_1_0fafbf1a-41c6-4cee-a0ea-44fec6fccc9b --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_3_0.2_256_512_256_4_4_0.0001_0.5_0.01_0.1_1_1_f6a997c9-8aec-405f-b4a2-9b68df617804/ ./saved_model/
# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_3_0.2_256_512_256_4_4_0.0001_0.5_0.01_0.1_1_1_f6a997c9-8aec-405f-b4a2-9b68df617804 --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_4_0.15_256_512_512_4_4_0.0001_0.1_0.01_0.05_1_1_86c482e0-96ab-4328-a5a5-6f7c61bc5c58/ ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_qid_deepbkt_tiaocan_assist2009_42_4_0.15_256_512_512_4_4_0.0001_0.1_0.01_0.05_1_1_86c482e0-96ab-4328-a5a5-6f7c61bc5c58 --bz=64 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_augmentation_bayesian_v2_deepbkt_tiaocan_assist2009_42_0_0.3_256_512_256_4_4_0.0001_0.5_0.05_0.15_1_1_b1ab1af3-1551-4607-a345-49c8ab070714 ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v2_deepbkt_tiaocan_assist2009_42_0_0.3_256_512_256_4_4_0.0001_0.5_0.05_0.15_1_1_b1ab1af3-1551-4607-a345-49c8ab070714 --bz=32 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_difficulty_deepbkt_tiaocan_assist2009_3407_0_0.2_256_512_512_4_4_0.0001_0.0_0.025_0.2_1_1_dc11a4b1-6413-4f11-9df8-3cd74bebf8c8 ./saved_model/
# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_difficulty_deepbkt_tiaocan_assist2009_3407_0_0.2_256_512_512_4_4_0.0001_0.0_0.025_0.2_1_1_dc11a4b1-6413-4f11-9df8-3cd74bebf8c8 --bz=128 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev2.0/pykt-toolkit/examples/saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.25_512_512_512_4_3_0.0001_0.3_0.05_0.1_0_0 ./saved_model/
# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.25_512_512_512_4_3_0.0001_0.3_0.05_0.1_0_0 --bz=64 --fusion_type=late_fusion > b_pred.txt &


# python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_deepbkt_tiaocan_assist2009_42_0_0.2_256_256_512_4_4_0.0001_0.5_0.1_0.05_1_1_6be32ab3-7a96-40fb-a101-69b8787bd987 --bz=128 --fusion_type=late_fusion --use_wandb=0

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.2_256_256_512_4_4_0.0001_0.3_0.05_0.1_0_0 --bz=128 --fusion_type=late_fusion > b_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.2_256_256_512_4_4_0.0001_0.3_0.05_0.1_0_0 --bz=128 --fusion_type=late_fusion > b_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v2_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > b_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v3_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > b_pred.txt &

# scp -rP 51757 root@region-3.autodl.com:/root/autodl-nas/huangshuyan/shyann_dev/pykt-toolkit/examples/deepbkt_tiaocan_assist2009/assist2009_deepbkt_forgetting_deepbkt_tiaocan_assist2009_224_0_0.25_256_512_256_4_4_0.0001_0.2_0.1_0.05_1_1_146f45a1-2dba-4e88-be49-530977d08fe0 ./saved_model/
# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_deepbkt_tiaocan_assist2009_224_0_0.25_256_512_256_4_4_0.0001_0.2_0.1_0.05_1_1_146f45a1-2dba-4e88-be49-530977d08fe0 --bz=64 --fusion_type=late_fusion > b_pred.txt &


CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_dina_saved_model_224_0_0.2_256_512_512_4_2_0.0001_0.5_0.05_0.2_0_0 --bz=64 --fusion_type=late_fusion > dina_pred_v3.txt &

