python ../examples/data_preprocess.py --dataset_name assist2015 
python ../examples/wandb_dkt_train.py --use_wandb 0 --add_uuid 0 
python ../examples/wandb_predict.py --save_dir saved_model/assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_0_0 --use_wandb 0
python ../examples/wandb_eval.py --save_dir saved_model/assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_0_0 --use_wandb 0
