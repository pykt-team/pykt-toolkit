import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #dataset config
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--fold", type=int, default=0)

    # train config
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)

    #log config & save config
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_model")

    # model config
    parser.add_argument("--model_name", type=str, default="qikt_ncd")
    parser.add_argument("--emb_type", type=str, default="iekt")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--mlp_layer_num", type=int, default=2)

    # loss mode
    parser.add_argument("--loss_q_all_lambda", type=float, default=0)#a
    parser.add_argument("--loss_c_all_lambda", type=float, default=0)#b
    parser.add_argument("--loss_c_next_lambda", type=float, default=0)#c
    parser.add_argument("--loss_ncd", type=float, default=0)#ncd

    #output mode
    parser.add_argument("--output_q_all_lambda", type=float, default=0)#a
    parser.add_argument("--output_c_all_lambda", type=float, default=0)#b
    parser.add_argument("--output_c_next_lambda", type=float, default=0)#c
    parser.add_argument("--output_ncd", type=float, default=0)#ncd
    
    
    # use for ab study
    parser.add_argument("--ab_mode", type=str, default="b",help="one of ['b','b+a','b+c','b+a+c','b+irt','b+a+irt','b+c+irt','b+a+c+irt']")

    
    args = parser.parse_args()

    ## ab study config process
    ab_mode = [str(x) for x in args.ab_mode.split("+")]

    if "a" in ab_mode:
        args.output_q_all_lambda = 1
    if "b" in ab_mode:
        args.output_c_all_lambda = 1
    if "c" in ab_mode:
        args.output_c_next_lambda = 1
    if 'ncd' in ab_mode:
        args.output_ncd = 1
    
    if "irt" in ab_mode:
        args.output_mode = "an_irt"
    else:
        args.output_mode = "an"

    params = vars(args)

    # add some config to other_config
    remove_keys = ['ab_mode','output_mode'] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    for k in remove_keys:
        if k in params:
            other_config[k] = params[k]
            del params[k]
    params['other_config'] = other_config
    main(params)


'''
python wandb_qikt_ncd_train.py --ab_mode a+b+c+ncd

'''