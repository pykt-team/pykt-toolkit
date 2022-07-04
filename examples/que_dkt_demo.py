import os
import torch
from pykt.models.dkt_que import DKTQue
from pykt.datasets.que_data_loader import KTQueDataset
import json
import sys

dataset_name = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_config = json.load(open('../configs/data_config.json'))[dataset_name]
data_config

all_folds = set(data_config["folds"])


i = 0
train_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                        input_type=data_config["input_type"], folds=all_folds - {i}, 
                        concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

valid_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                        input_type=data_config["input_type"], folds={i}, 
                        concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                        input_type=data_config["input_type"], folds=[-1], 
                        concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

model = DKTQue(num_q=data_config['num_q'], num_c=data_config['num_c'],
               emb_size=300, device=device, emb_type='qid',seed=3407)

model.compile(optimizer='adam', lr=0.001)

model.train(train_dataset, valid_dataset, batch_size=32,
            num_epochs=200, patient=10, shuffle=False,save_model=True,save_dir='tmp/'+dataset_name)

model.load_model(model.save_dir)
valid_result = model.evaluate(valid_dataset,batch_size=32)
print(f"valid_result is {valid_result}")

test_result = model.evaluate(test_dataset,batch_size=32)
print(f"test_result is {valid_result}")
