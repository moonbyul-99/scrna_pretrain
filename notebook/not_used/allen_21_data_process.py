import numpy as np 
import pandas as pd 
import os 
import h5py
from tqdm import tqdm
import pyarrow
from datasets import load_dataset 
import torch 
import torch.nn as nn 
import time 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import torch.nn.functional as F
 
    
if __name__ == '__main__':
    batch_size = 10000
    
    dataset_dir = '/work/sunrui/pretrain_dataset'
    
    data_path = os.path.join(dataset_dir, 'allen_2021_data')
    
    if not os.path.exists(data_path):
        os.makedirs(data_path) 
        
    f = h5py.File("/work/sunrui/pretrain_dataset/allen_2021_ori_data/CTX_Hip_counts_10x.h5",'r') 
    
    sample = f['data']['samples'][:].astype('U')
    
    L = int(sample.shape[0] // batch_size) + 1
    
    gene_id_identifier = 'mus_gene_1'
    data_source = 'allen_21_mus_brain_scrna'
    
    for i in tqdm(range(L)):
        
        begin = batch_size*i 
        end = min(batch_size*(i+1), sample.shape[0])

        counts = f['data']['counts'][:,begin:end].T
        cell_ids = sample[begin:end]
        
        N = counts.shape[0] 
        data_dic_list = []
        for k in range(N):
            count = counts[k,:] 
            
            meta_info = {'cell_id': cell_ids[k],
                         'data_source': data_source}
            data_dic = {'count': count,
                        'gene_id_identifier': gene_id_identifier, 
                        'meta_info': meta_info}
            data_dic_list.append(data_dic)
            

        df = pd.DataFrame(data_dic_list)
        
        if i < 10:
            file_name = f'part_00{i}.parquet'
        elif i < 100:
            file_name = f'part_0{i}.parquet' 
        else:
            file_name = f'part_{i}.parquet' 
            
        output_file = os.path.join(data_path, file_name)

        df.to_parquet(output_file, engine = 'pyarrow')
    
    print('data split is over')

    dataset = load_dataset(path = data_path , cache_dir = '/work/sunrui/huggingface')
    
    
    
    
    
    