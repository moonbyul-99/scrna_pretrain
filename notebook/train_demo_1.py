import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim 
from datasets import load_dataset 

import os 
import h5py
from tqdm import tqdm
import pyarrow
import time 
import json 

import sys 
sys.path.append('../code')
import utils
import model 
import loss 
from custom_dataset import CustomDataset
import train_v1 as train


if __name__ == '__main__':
    (gene_dict, dataset_gene, dataset_gene_ids) = utils.generate_gene_dic()

    tokenizer = utils.tokenizer_v1(gene_dict= gene_dict,
                             dataset_gene= dataset_gene,
                             dataset_gene_ids= dataset_gene_ids) 

    vocab_size = tokenizer.vocab_size
    print(vocab_size)

    tokenizer.add_token(token = '<cls>')
    tokenizer.add_token(token = '<pad>')
    #tokenizer.gene_dict['<cls>'] = vocab_size
    print(tokenizer.vocab_size)
    print(tokenizer.gene_dict['<cls>']) 
    print(tokenizer.gene_dict['<pad>'])  
    
    print('tokenization over')
    collate_fn = utils.collater(tokenizer= tokenizer, max_expression= 100, mask_ratio = 0.2, 
                                max_num = 2000,  rho = 0.1, pad_idx = tokenizer.gene_dict['<pad>'])
    
    

    #dataset_1  = load_dataset(path = 'mus_brain', cache_dir = 'huggingface_cache')
    dataset_1 = load_dataset(path = '/work/sunrui/pretrain_dataset/allen_2021_data',
                             cache_dir = '/work/sunrui/huggingface')
    dataset_2 = load_dataset(path = '/work/sunrui/pretrain_dataset/allen_2023_data', 
                             cache_dir = '/work/sunrui/huggingface') 
    
    print('data is load')
    dataset_1 = dataset_1['train'].train_test_split(test_size = 0.05)
    dataset_2= dataset_2['train'].train_test_split(test_size = 0.05)
    print('data is split')
    
    train_dataset_1, test_dataset_1 = dataset_1['train'], dataset_2['test']
    train_dataset_2, test_dataset_2 = dataset_2['train'], dataset_2['test']

    train_dataset = CustomDataset([train_dataset_1, train_dataset_2]) 
    test_dataset = CustomDataset([test_dataset_1, test_dataset_2]) 
    
    #print(len(train_dataset), len(test_dataset))
    print('dataset is over')
    
    count_embedding_num = 104
    gene_embedding_num = tokenizer.vocab_size

    d_model = 256
    gene_padding_idx = tokenizer.gene_dict['<pad>']
    count_padding_idx = 103
    n_head = 8
    dim_ffn = 4*d_model
    dropout = 0.1
    layer_norm_eps =1e-5
    batch_first = True
    norm_first = False
    num_layers = 8
    norm = None
    num_hiddens = 256

    my_model = model.sc_pretrain(count_embedding_num,
                     gene_embedding_num,
                     d_model,
                     gene_padding_idx,
                     count_padding_idx,
                     n_head,
                     dim_ffn,
                     dropout,
                     layer_norm_eps,
                     batch_first,
                     norm_first,
                     num_layers,
                     norm,
                     num_hiddens) 
 

    # 创建 DataLoader 实例
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn= collate_fn)

    pretrain_loss = loss.pretrain_loss()

    my_model = train.train_multi_epoch(my_model, 
                    train_loader,
                    test_loader,
                    pretrain_loss, 
                    #optimizer = optim.SGD(my_model.parameters(), lr=1e-4, momentum=0.9),
                   optimizer = optim.Adam(my_model.parameters(), lr = 5e-5, weight_decay=0.01),
                    device = 'cuda',
                    gradient_accumulation_steps = 32,
                    save_steps = 7500,
                    save_dir = 'test_2',
                    epochs = 5)
    
    print('train phase over')
    
    
    