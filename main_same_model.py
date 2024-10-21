# THis is to run same model for both the datasets and all the configurations . For this you need to run the training on 50 Salads and model with run breakfsat by its own and train on both the datasets.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import pdb
import random
from torch.backends import cudnn
from opts import parser
from scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR

from utils import read_mapping_dict
from data.basedataset import BaseDataset
from model.futr import FUTR
from train import train
from predict import predict
import math
from torch.utils.data import DataLoader, ConcatDataset


device = torch.device('cuda')

# Seed fix
# seed = 13452  (This seed is for sure bad for our networkd)
# seed = 42
# seed = 26
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# cudnn.benchmark, cudnn.deterministic = False, True


def get_lr_lambda(warmup_epochs, total_epochs, min_lr=0.0):
    """Create a learning rate lambda function for Linear Warmup and Cosine Annealing schedule"""
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda

def main():
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')

    print('runs : ', args.runs)
    print('model type : ', args.model)
    print('input type : ', args.input_type)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    dataset = args.dataset
    task = args.task
    split = args.split

    if dataset == 'breakfast':
        data_path = './datasets/breakfast'
    elif dataset == '50salads' :
        data_path = './datasets/50salads'

    mapping_file = os.path.join("./datasets/mapping_all.txt")
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train.split'+args.split+'.bundle' )
    video_file_test_path = os.path.join(data_path, 'splits', 'test.split'+args.split+'.bundle' )

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features')
    gt_path = os.path.join(data_path, 'groundTruth')

    n_class = len(actions_dict) + 1
    pad_idx = n_class + 1

    finetune = True if args.finetune else False

    # Model specification
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    # model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/transformer', split, args.input_type, \
    #                                 'runs'+str(args.runs))
    model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/diffusion_all', split, args.input_type, \
                                    'runs'+str(args.runs))
    results_save_path = os.path.join('./save_dir/'+args.dataset+'/'+args.task+'/results/transformer', 'split'+split,
                                    args.input_type )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)


    model_save_file = os.path.join(model_save_path, 'checkpoint.ckpt')
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    
    criterion = nn.MSELoss(reduction = 'none')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


    if args.predict:
        if args.demo_predict:
            import glob
            video_test_list = glob.glob(args.demo_data_path + 'videos/*avi')
            video_test_list = [vid.split('/')[-1] for vid in video_test_list]

        # obs_perc = [0.05, 0.1, 0.2, 0.3]
        obs_perc = [0.2, 0.3]
        # obs_perc = [0.05, 0.1]
        results_save_path = results_save_path +'/runs'+ str(args.runs) +'.txt'
        if args.dataset == 'breakfast' :
            # model_path = './ckpt/bf_split'+args.split+'.ckpt'
            # model_path = './ckpt_test/bf_split'+args.split+'.ckpt'
            # model_path = './ckpt_pretrained/bf_split'+args.split+'.ckpt'
            # model_path = "/data/aryan/Seekg/knowledge_diffusion/save_dir/breakfast/long/model/diffusion_step_analysis/1/i3d_transcript/runs2/checkpoint37.ckpt"
            models_path = "./save_dir/50salads/long/model/diffusion_all/1/i3d_transcript/runs2"
            models_path = [os.path.join(models_path,model) for model in sorted(os.listdir(models_path)) if "checkpoint" in model]
        elif args.dataset == '50salads':
            # model_path = './ckpt/50s_split'+args.split+'.ckpt'
            # model_path = './ckpt_pretrained/50s_split'+args.split+'.ckpt'
            # model_path = "/data/aryan/Seekg/knowledge_diffusion/save_dir/50salads/long/model/diffusion_all/1/i3d_transcript/runs2/checkpoint45.ckpt"
            # model_path = '/data/aryan/Seekg/FUTR/save_dir/50salads/long/model/diffusion/2/i3d_transcript/mamba_transformer_st/checkpoint40.ckpt'
            models_path = "./save_dir/50salads/long/model/diffusion_all/1/i3d_transcript/runs2"
            # models_path = "./save_dir/50salads/long/model/transformer/1/i3d_transcript/runs0"
            models_path = [os.path.join(models_path,model) for model in sorted(os.listdir(models_path)) if "checkpoint" in model]
        # print("Predict with ", model_path)


        # for obs_p in obs_perc :
        #     model.load_state_dict(torch.load(model_path))
        #     model.to(device)
        #     predict(model, video_test_list, args, obs_p, n_class, actions_dict, device)

        for model_path in models_path :
            print("Predict with ", model_path)
            for obs_p in obs_perc :
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                predict(model, video_test_list, args, obs_p, n_class, actions_dict, device)
    else :
        if args.finetune:
            if args.dataset == 'breakfast':
                model_path = './ckpt/bf_split'+args.split+'.ckpt'
            elif args.dataset == '50salads':
                model_path = './ckpt/50s_split'+args.split+'.ckpt'
            print("Loading model from ", model_path)

            model.load_state_dict(torch.load(model_path))
            model.to(device)

        # Load the model weihts from the models not trained on any graph based setting.
        # if dataset == 'breakfast':
        #     saved_model_path = './ckpt_pretrained/bf_split'+args.split+'.ckpt'
        # elif dataset == '50salads':
        #     saved_model_path = './ckpt_pretrained/50s_split'+args.split+'.ckpt'
        # saved_model_state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
        # model_state_dict = model.state_dict()
        # new_model_state_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_state_dict}
        # model.load_state_dict(new_model_state_dict, strict=False)
        # # model.load_state_dict(saved_model_state_dict)



        # Training
        trainset_1 = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, finetune=finetune)
        # train_loader_1 = DataLoader(trainset_1, batch_size=args.batch_size, \
        #                                             shuffle=True, num_workers=args.workers,
        #                                             collate_fn=trainset_1.my_collate)
        
        data_path = './datasets/breakfast'
        video_file_path = os.path.join(data_path, 'splits', 'train.split'+args.split+'.bundle' )
        video_file = open(video_file_path, 'r')
        video_list = video_file.read().split('\n')[:-1]
        features_path = os.path.join(data_path, 'features')
        gt_path = os.path.join(data_path, 'groundTruth') 
        args.dataset = 'breakfast'

        trainset_2 = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, finetune=finetune)
        # train_loader_2 = DataLoader(trainset_2, batch_size=args.batch_size, \
        #                                             shuffle=True, num_workers=args.workers,
        #                                             collate_fn=trainset_2.my_collate)

        combined_dataset = ConcatDataset([trainset_1, trainset_2])
        
        train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset_2.my_collate)



        train(args, model,train_loader , optimizer, scheduler, criterion,
                     model_save_path, pad_idx, actions_dict, device )


if __name__ == '__main__':
    main()
