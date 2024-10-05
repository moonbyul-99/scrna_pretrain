import numpy as np 
import pandas as pd 
import os 
import json 
import torch 

def generate_gene_dic():
    mus_gene_0 = np.load('../gene_dict/mus_gene_0.npy',allow_pickle=True)
    mus_gene_1 = np.load('../gene_dict/mus_gene_1.npy',allow_pickle=True)

    mus_gene_0_id = np.load('../gene_dict/mus_gene_id_0.npy',allow_pickle=True)
    mus_gene_1_id = np.load('../gene_dict/mus_gene_id_1.npy',allow_pickle=True)

    dataset_gene = {}

    dataset_gene['mus_gene_0'] = list(mus_gene_0) 
    dataset_gene['mus_gene_1'] = list(mus_gene_1) 

    dataset_gene_ids = {}

    dataset_gene_ids['mus_gene_0'] = list(map(int,mus_gene_0_id))
    dataset_gene_ids['mus_gene_1'] = list(map(int,mus_gene_1_id))

    with open('../gene_dict/dataset_gene_info.json','w') as f:
        json.dump(dataset_gene,f) 

    with open('../gene_dict/dataset_gene_ids_info.json','w') as f:
        json.dump(dataset_gene_ids,f)

    # 指定json文件路径
    file_path = '../gene_dict/gene_dict.json'

    # 使用with语句打开文件，这样可以自动管理文件关闭
    with open(file_path, 'r', encoding='utf-8') as file:
        # 使用json.load()方法加载文件内容到字典
        gene_dict = json.load(file)

    for key in dataset_gene:
        dataset_gene[key] = np.array(dataset_gene[key])

    for key in dataset_gene_ids:
        dataset_gene_ids[key] = np.array(dataset_gene_ids[key])
    
    return (gene_dict, dataset_gene, dataset_gene_ids)

def gene_sample_1(count, max_num, rho = 0.1, pad_idx = -1):
    # sample stratege 1:
    # for cells with expressed gene num greater than max_num, random sample rho*max_num expressed gene and (1-rho)*max_num not expressed gene
    # for cells with expressed gene num less than max_num, take all expressed gene and min(max_num - L_pos, L_pos) not expressed gene 

    # the probability of non-expressed gene being sampled is zero if based on counts
    # provide a pseudo count, such that the non-expressed gene being sampled is rho  
    # we have (N_neg*\eta)/(umi_count + N*\eta) = rho 
    # then we have \eta = rho*umi_count / (N_neg - rho*N)
    

    # rho: the probability of zero-count genes being sampled, float 0-1 , default = 0.1
    # return sample ids 

    if isinstance(count, torch.Tensor) :
        count = count.numpy()
    if isinstance(count, list):
        count = np.array(count)
    umi_count = count.sum()
    N_neg = (count == 0).sum()
    N = count.shape[0]
    eta = max( (rho*umi_count) / (N_neg - rho*N) , 0) 

    sample_prob = (count + eta)/ (umi_count + N*eta)

    sample_ids = np.random.choice(np.arange(N), size = max_num, replace=False, p = sample_prob)
    return sample_ids


class tokenizer_v1:
    def __init__(self,gene_dict, dataset_gene, dataset_gene_ids):
        self.gene_dict = gene_dict 
        self.dataset_gene = dataset_gene
        self.dataset_gene_ids = dataset_gene_ids
        #self.vocab_size = len(gene_dict)

    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return len(self.gene_dict)


    def add_token(self, token, index = None):
        if index is None:
            index = self.vocab_size
        if token not in self.gene_dict:
            if index not in self.gene_dict.values():
                self.gene_dict[token] = index
            else:
                raise ValueError("index already exists")
        


    def get_token_id(self, ids, gene_id_identifier):
        # ids is a list
        return self.dataset_gene_ids[gene_id_identifier][ids]
    
    def get_token_name(self, ids, gene_id_identifier):
        return self.dataset_gene[gene_id_identifier][ids] 



class collater():
    def __init__(self, tokenizer, max_expression, mask_ratio, max_num, rho, pad_idx = -1):
        self.tokenizer = tokenizer 
        self.max_num = max_num 
        self.rho = rho 
        self.pad_idx = pad_idx
        self.max_expression = max_expression
        self.mask_ratio = mask_ratio
        self.gene_cls_id = tokenizer.gene_dict['<cls>']
        self.count_mask_id = max_expression + 1 
        self.count_cls_id = max_expression + 2
        #self.mask_id = tokenizer.gene_dict['<mask>']

    def __call__(self, batch):
        batch_data = {} 

        batch_data['counts_0'] = []
        batch_data['counts_1'] = []

        batch_data['token_id_0'] = []
        batch_data['token_id_1'] = [] 

        batch_data['mask_array'] = []

        for sample in batch:
            count = np.array(sample['count'])
            gene_id_identifier = sample['gene_id_identifier'] 

            # perform down sample
            sample_ids_0 = gene_sample_1(count, max_num = self.max_num, rho = self.rho, pad_idx= self.pad_idx)
            sample_ids_1 = gene_sample_1(count, max_num = self.max_num, rho = self.rho, pad_idx= self.pad_idx)

            # encode gene_id towards number id 
            token_id_0 = self.tokenizer.get_token_id(ids = sample_ids_0 ,gene_id_identifier= gene_id_identifier)
            token_id_1 = self.tokenizer.get_token_id(ids = sample_ids_1 ,gene_id_identifier= gene_id_identifier) 

            # clip the counts 

            counts_0 = np.clip(count[sample_ids_0], a_min = 0, a_max = self.max_expression)
            counts_1 = np.clip(count[sample_ids_1], a_min = 0, a_max = self.max_expression) 

            # mask the counts 
            mask_array = np.random.choice([True, False], size= self.max_num, p=[self.mask_ratio, 1-self.mask_ratio]) 

            # add the cls token in counts and token_id

            #counts_0[mask_array] = self.count_mask_id 
            #counts_1[mask_array] = self.count_mask_id 


            token_id_0 = np.insert(token_id_0, 0, self.gene_cls_id)
            token_id_1 = np.insert(token_id_1, 0, self.gene_cls_id) 

            counts_0 = np.insert(counts_0, 0, self.count_cls_id) 
            counts_1 = np.insert(counts_1, 0, self.count_cls_id) 

            mask_array = np.insert(mask_array, 0, False) 

            # add to batch_data 
            batch_data['counts_0'].append(counts_0)
            batch_data['counts_1'].append(counts_1)
            batch_data['token_id_0'].append(token_id_0)
            batch_data['token_id_1'].append(token_id_1)
            batch_data['mask_array'].append(mask_array)


        batch_data['counts_0'] = torch.tensor(batch_data['counts_0'], dtype = torch.int)
        batch_data['counts_1'] = torch.tensor(batch_data['counts_1'], dtype = torch.int) 

        batch_data['counts_0'] = batch_data['counts_0'].long()
        batch_data['counts_1'] = batch_data['counts_1'].long()

        batch_data['token_id_0'] = torch.tensor(batch_data['token_id_0'], dtype = torch.int)
        batch_data['token_id_1'] = torch.tensor(batch_data['token_id_1'], dtype = torch.int)
        batch_data['mask_array'] = torch.tensor(batch_data['mask_array'], dtype = torch.bool) 

        counts = torch.cat((batch_data['counts_0'], batch_data['counts_1']))
        token_id = torch.cat((batch_data['token_id_0'], batch_data['token_id_1']))
        mask_array = torch.cat((batch_data['mask_array'], batch_data['mask_array'])) 
        label = counts[mask_array]
        counts[mask_array] = self.count_mask_id
        
        return (counts, token_id, mask_array, label)
    
