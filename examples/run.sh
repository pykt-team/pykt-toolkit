#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=relation --d_ff=512 --n_blocks=4 --dropout=0.2 --use_wandb=0 --add_uuid=0 > akt_relation.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=lstm --use_wandb=0 --add_uuid=0 > akt_lstm.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=lstmy --use_wandb=0 --add_uuid=0 > akt_lstmy.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=lstmy_bayesian --use_wandb=0 --add_uuid=0 > akt_lstmy_bayesian.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=relation_lstmy_bayesian --use_wandb=0 --add_uuid=0 > akt_relation_lstmy_bayesian.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=bayesian_loss --use_wandb=0 --add_uuid=0 > akt_bayesian_loss.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=yplus_kc --use_wandb=0 --add_uuid=0 > akt_yplus_kc_v2.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=yplus_que --use_wandb=0 --add_uuid=0 > akt_yplus_que.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=bernoulli --use_wandb=0 --add_uuid=0 > akt_bayesian_scalar.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_akt_vector_relation_bernoulli_saved_model_42_0_0.05_256_256_8_1_1e-05_0_0 > akt_relation_bernoulli.pred &
# CUDA_VISIBLE_DEVICES=2 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --save_dir &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=raschy --use_wandb=0 --add_uuid=0 > akt_raschy.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=relation --use_wandb=0 --add_uuid=0 > akt_relation.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=relation_bayesian --use_wandb=0 --add_uuid=0 > akt_relation_bayesian_scalar.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=relation_bayesian_loss --use_wandb=0 --add_uuid=0 > akt_relation_bayesian_loss.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_skvmn_train.py --dataset_name=statics2011 --use_wandb=0 --add_uuid=0 > skvmn.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_forget_train.py --dataset_name=assist2009 --emb_type=atc --use_wandb=0 --add_uuid=0 > akt_atc_v2.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_train.py --dataset_name=assist2009 --emb_type=yplus_kc --use_wandb=0 --add_uuid=0 > akt_yplus_kc_novec.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 > akt_vector_mysig.txt &


# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 > akt_original_v2.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_perturbation_train.py --dataset_name=assist2009 --emb_type=perturbation --use_wandb=0 --add_uuid=0 > akt_perturbation.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_aktforget_train.py --dataset_name=assist2009 --emb_type=ratio --use_wandb=0 --add_uuid=0 > aktforget.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_train.py --dataset_name=assist2009 --use_wandb=0 --add_uuid=0 > akt.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_forget_train.py --dataset_name=assist2009 --emb_type=concat --use_wandb=0 --add_uuid=0 > akt_forget_concat.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_perturbation_train.py --dataset_name=assist2009 --emb_type=perturbation_bayesian --use_wandb=0 --add_uuid=0 > akt_forget_perturbation_bayesian.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_forget_train.py --dataset_name=assist2009 --emb_type=inner_bayesian --use_wandb=0 --add_uuid=0 > akt_forget_inner_bayesian.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 > akt_vector_bayesian_plus.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_aktforget_train.py --dataset_name=assist2009 --emb_type=mforget --use_wandb=0 --add_uuid=0 > akt_mforget.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 > deepbkt.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 > akt_vector_bayesian_plus.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting --use_wandb=0 --add_uuid=0 --d_ff=256 --d_model=256  --dropout=0.2 --final_fc_dim=512 --seed=42 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 > deepbkt_sigmoidbest.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=/data/huangshuyan/shyann_dev2.0/pykt-toolkit/examples/saved_model/assist2009_deepbkt_forgetting_saved_model_3407_0_0.25_512_512_512_4_3_0.0001_0.3_0.05_0.1_0_0 --bz=128 --fusion_type=late_fusion > ./pred/new_pred.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_bayesian_v2_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > ./pred/deepbkt_bayesian_plus_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v2 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian_plus.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v4 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian_plus_plus_v4.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting --use_wandb=0 --add_uuid=0 --d_ff=256 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --seed=42 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.1 --sigmoidb=0.05 > pred/deepbkt_forgetting.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_bayesian_v2_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > ./pred/deepbkt_bayesian_plus_pred.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.2_256_256_512_4_4_0.0001_0.5_0.1_0.05_0_0 --bz=64 --fusion_type=late_fusion > ./pred/deepbkt_forgetting_pred.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_bayesian_v3_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > ./pred/deepbkt_bayesian_v3_pred.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_bayesian_v4_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=128 --fusion_type=late_fusion > ./pred/deepbkt_bayesian_v4_pred.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_predict.py --save_dir=./saved_model/assist2009_deepbkt_difficulty_v2_saved_model_3407_0_0.2_256_512_512_4_4_0.0001_0.3_0.025_0.2_0_0 --bz=128 --fusion_type=late_fusion > ./pred/deepbkt_difficulty_v2.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=difficulty_v2 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=3407  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.025 --sigmoidb=0.2 > pred/deepbkt_difficulty_v2.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_v3 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_augmentation_v3.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_bayesian_v2 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_augmentation_bayesian_v2.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_bayesian_v3 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_augmentation_bayesian_v3.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting_v3 --use_wandb=0 --add_uuid=0 --d_ff=256 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --seed=42 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.1 --sigmoidb=0.05 > pred/deepbkt_forgetting_v3.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_v3_saved_model_42_0_0.2_256_256_512_4_4_0.0001_0.5_0.1_0.05_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_forgetting_v3_pred.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v3_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_augmentation_bayesian_v3_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=4 --seed=224 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_qid.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v3 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian_v3.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v2 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_bayesian_v2.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=4 --seed=224 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_bayesian.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=4 --seed=224 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_bayesian2.0.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=3 --num_attn_heads=4 --seed=224 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.1 --sigmoidb=0.1 > pred/deepbkt_augmentation_bayesian.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_bayesian_v2 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=3 --num_attn_heads=4 --seed=224 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.1 --sigmoidb=0.1 > pred/deepbkt_augmentation_bayesian_v2.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=augmentation_bayesian_v3 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=3 --num_attn_heads=4 --seed=224 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.1 --sigmoidb=0.1 > pred/deepbkt_augmentation_bayesian_v3.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_saved_model_224_0_0.2_256_512_256_4_3_0.0001_0.5_0.1_0.1_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_augmentation_augmentation_bayesian_v3_pred.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v2_saved_model_224_0_0.2_256_512_256_4_3_0.0001_0.5_0.1_0.1_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_augmentation_augmentation_bayesian_v2_pred.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_augmentation_bayesian_v3_saved_model_224_0_0.2_256_512_256_4_3_0.0001_0.5_0.1_0.1_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_augmentation_augmentation_bayesian_pred.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_forgetting_v2.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_predict.py --save_dir=saved_model/assist2009_deepbkt_forgetting_saved_model_42_0_0.2_256_512_512_4_4_0.0001_0.3_0.01_0.2_0_0 --bz=64 --fusion_type=late_fusion > deepbkt_forgetting_pred_v2.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_akt_train.py --dataset_name=assist2009 --emb_type=forget --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=3407  --dropout=0.2 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=8 > pred/akt_forget.txt &


# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v4 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=512 --dropout=0.25 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=8 --seed=42 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_bayesian_v4.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v5 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=512 --dropout=0.3 --learning_rate=0.0001 --n_blocks=1 --num_attn_heads=8 --seed=224 --final_fc_dim=256 --lambda_r=0.1 --sigmoida=0.1 --sigmoidb=0.025 > pred/deepbkt_bayesian_v5.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_forgetting_v3.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.25 --learning_rate=0.0001 --n_blocks=3 --num_attn_heads=4 --seed=3407 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_bayesian_relu.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=dina --use_wandb=0 --add_uuid=0 --d_ff=256 --d_model=512 --dropout=0.2 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=8 --seed=3407 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > sep_pred/dina.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=dina_v3 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=4 --seed=224 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > sep_pred/deepbkt_dina_v6.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.2 --learning_rate=0.0001 --n_blocks=2 --num_attn_heads=4 --seed=224 --final_fc_dim=512 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > sep_pred/deepbkt_qid_v2.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=256 --d_model=512 --dropout=0.2 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=8 --seed=3407 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > sep_pred/deepbkt_bayesian.txt &

# CUDA_VISIBLE_DEVICES=0 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian_v5 --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=512 --dropout=0.3 --learning_rate=0.0001 --n_blocks=1 --num_attn_heads=8 --seed=224 --final_fc_dim=256 --lambda_r=0.1 --sigmoida=0.1 --sigmoidb=0.025 > pred/deepbkt_bayesian_v5.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=forgetting --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --seed=42  --dropout=0.2 --final_fc_dim=512 --learning_rate=0.0001 --n_blocks=4 --num_attn_heads=4 --sigmoida=0.01 --sigmoidb=0.2 > pred/deepbkt_forgetting_v3.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 --d_ff=512 --d_model=256 --dropout=0.25 --learning_rate=0.0001 --n_blocks=3 --num_attn_heads=4 --seed=3407 --final_fc_dim=256 --lambda_r=0.5 --sigmoida=0.05 --sigmoidb=0.2 > pred/deepbkt_bayesian_relu.txt &