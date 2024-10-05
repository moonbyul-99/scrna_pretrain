import torch
import torch.nn as nn 
import torch.nn.functional as F


def cosine_sim_loss(cell_1, cell_2):
    # cell_1
    #similarity_matrix = torch.matmul(cell_1, cell_2)
    loss = F.cosine_similarity(cell_1, cell_2).mean()
    return loss

def gene_exp_crossentropy(gene_exp, target_exp):
    return F.cross_entropy(input = gene_exp, target = target_exp, reduction='mean')

class CLIPLoss(torch.nn.Module):
    """
    Implementation of the CLIP loss function.
    """
    
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, cell_1, cell_2):
        # 特征向量归一化
        cell_1 = F.normalize(cell_1, dim=-1)
        cell_2 = F.normalize(cell_2, dim=-1)

        # 计算相似度矩阵
        logits_cell_1 = torch.div(
            torch.matmul(cell_1, cell_2.t()),
            self.temperature
        )
        logits_cell_2 = logits_cell_1.t()
        
        # 计算图像到文本的损失
        ground_truth = torch.arange(len(cell_1), dtype=torch.long, device=cell_1.device)
        loss_12 = F.cross_entropy(logits_cell_1, ground_truth)
        
        # 计算文本到图像的损失
        loss_21 = F.cross_entropy(logits_cell_2, ground_truth)
        
        # 返回总损失
        return (loss_12 + loss_21) / 2

class pretrain_loss(torch.nn.Module):
    def __init__(self, alpha = 1, temperature = 0.07):
        super(pretrain_loss, self).__init__()

        self.temperature = temperature 
        self.alpha = alpha 

        self.clip_loss = CLIPLoss(temperature=self.temperature)
        return None 
    
    def forward(self, cell_1,cell_2, gene_exp, target_exp):
        exp_loss = gene_exp_crossentropy(gene_exp, target_exp)
        clip_loss = self.clip_loss(cell_1,cell_2)
        return {'total_loss': exp_loss + self.alpha*clip_loss, 
                'gene_exp_ce_loss':exp_loss, 
                'clip_loss':clip_loss}
        
