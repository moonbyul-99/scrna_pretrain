import numpy as np 
import pandas as pd 
import os 
import h5py
from tqdm import tqdm
import pyarrow
import time 
import scanpy as sc
 


def get_allen_23_data():
    file_paths = []
    for root, dirs, files in os.walk('/work/sunrui/allen_whole_brain'):
        for filename in files:
            filepath = os.path.join(root, filename)
            if 'AIT17.0' in filepath:
                continue
            file_paths.append(filepath)
    return file_paths
    
    
def scdata_process(scdata, save_dir, gene_id_identifier = 'mus_gene_0', 
                  data_source = 'allen_23_mus_brain_scrna',
                  batch_size = 10000):
    
    total_counts = np.array(scdata.X.todense()).astype(int)
    total_ids = scdata.obs.index.values 
    total_meta_info = scdata.obs.copy()
    
    
    L = int(scdata.shape[0]// batch_size) + 1
    
    for i in tqdm(range(L)):
        
        begin = batch_size*i 
        end = min(batch_size*(i+1), total_counts.shape[0]) 
        
        counts = total_counts[begin:end,:]
        cell_ids = total_ids[begin:end]
        cell_meta_info = total_meta_info.iloc[begin:end,:]
    
        N = counts.shape[0] 
        data_dic_list = []
        for k in range(N):
            count = counts[k,:] 
            
            meta_info = dict(cell_meta_info.iloc[k,:])
            meta_info['cell_id'] =  cell_ids[k]
            meta_info['data_source'] = data_source
            
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
            
        output_file = os.path.join(save_dir, file_name)

        df.to_parquet(output_file, engine = 'pyarrow')
        
    print('data split is over')
    return None
    

if __name__ == '__main__':
    
    dataset_dir = '/work/sunrui/pretrain_dataset/allen_2023_data'
    file_paths = get_allen_23_data()
    
    for i,file in enumerate(file_paths):
        
        scdata = sc.read_h5ad(file)
        
        info = file.split('/')[-1].split('-')
        sub_dir = '-'.join(info[1:-1]) 
        
        save_dir = os.path.join(dataset_dir,sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f'{i}-th data {sub_dir} begin processing')
        
        scdata_process(scdata, 
                       save_dir = save_dir, 
                       gene_id_identifier = 'mus_gene_0', 
                      data_source = 'allen_23_mus_brain_scrna',
                      batch_size = 10000)
    print('allen_2023_data is over') 
    #dataset = load_dataset(path = dataset_dir , cache_dir = '/work/sunrui/huggingface')
            
        
    
    
    
    