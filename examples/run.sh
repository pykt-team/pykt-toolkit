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

CUDA_VISIBLE_DEVICES=2 nohup python wandb_deepbkt_train.py --dataset_name=assist2009 --emb_type=qid --use_wandb=0 --add_uuid=0 > deepbkt.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python wandb_akt_vector_train.py --dataset_name=assist2009 --emb_type=bayesian --use_wandb=0 --add_uuid=0 > akt_vector_bayesian_plus.txt &
