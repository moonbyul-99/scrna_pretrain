import torch 
import torch.nn as nn 

class count_embedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings= num_embeddings,
                                      embedding_dim= embedding_dim,
                                      padding_idx= padding_idx)
        return None 
    
    def forward(self, x):
        return self.embedding(x)
    

class gene_embedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings= num_embeddings,
                                      embedding_dim= embedding_dim,
                                      padding_idx= padding_idx)
        return None 
    
    def forward(self, x):
        return self.embedding(x)
    
class embedding_layer(nn.Module):
    def __init__(self,
                 count_embedding_num,
                 gene_embedding_num,
                 embedding_dim,
                 gene_padding_idx,
                 count_padding_idx
                 ):
        super().__init__()
        self.count_embedding = count_embedding(num_embeddings= count_embedding_num,
                                               embedding_dim= embedding_dim,
                                               padding_idx= count_padding_idx)
        
        self.gene_embedding = gene_embedding(num_embeddings= gene_embedding_num,
                                             embedding_dim= embedding_dim,
                                             padding_idx= gene_padding_idx)

        return None

    def forward(self, gene_id, count_id):
        return self.gene_embedding(gene_id) + self.count_embedding(count_id)
    

class transformer_model(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 layer_norm_eps,
                 batch_first,
                 norm_first,
                 num_layers,
                 norm):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                        nhead = nhead,
                                                        dim_feedforward= dim_ffn,
                                                        dropout= dropout,
                                                        activation= 'relu',
                                                        layer_norm_eps= layer_norm_eps, 
                                                        batch_first= batch_first,
                                                        norm_first= norm_first)
        
        self.encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer, 
                                             num_layers = num_layers,
                                             norm= norm)

        return None 
    
    def forward(self, src, mask):
        x = self.encoder(src, src_key_padding_mask = mask, is_causal = False)
        return x 
    
class mask_gp(nn.Module):
    # mask gene prediction 
    def __init__(self,
                 d_model,
                 num_hiddens,
                 count_embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, count_embedding_dim)
        )
        return None
    
    def forward(self, x, pred_positions):
        #num_pred_positions = pred_positions.shape[1]
        #pred_positions = pred_positions.reshape(-1)

        #batch_size = x.shape[0]
        #batch_idx = torch.arange(0, batch_size) 

        #batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_x = x[pred_positions,:]
        #masked_x = masked_x.reshape([batch_size, num_pred_positions, -1])
        mgp = self.mlp(masked_x)
        return mgp 
    
class cell_encoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_hiddens):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_hiddens)
        )
        return None 
    
    def forward(self,x):
        return self.mlp(x)
    
class sc_pretrain(nn.Module):
    def __init__(self,
                 count_embedding_num,
                 gene_embedding_num,
                 d_model,
                 gene_padding_idx,
                 count_padding_idx,
                 nhead,
                 dim_ffn,
                 dropout,
                 layer_norm_eps,
                 batch_first,
                 norm_first,
                 num_layers,
                 norm,
                 num_hiddens):
        super().__init__()

        self.embedding_layer = embedding_layer(count_embedding_num= count_embedding_num,
                                               gene_embedding_num= gene_embedding_num,
                                               embedding_dim = d_model,
                                               gene_padding_idx= gene_padding_idx,
                                               count_padding_idx= count_padding_idx
                                               )
        
        self.sc_encoder = transformer_model(d_model = d_model,
                                            nhead = nhead,
                                            dim_ffn = dim_ffn,
                                            dropout= dropout,
                                            layer_norm_eps= layer_norm_eps,
                                            batch_first= batch_first,
                                            norm_first= norm_first,
                                            num_layers= num_layers,
                                            norm= norm) 
        
        self.mask_gp = mask_gp(d_model = d_model,
                               num_hiddens = num_hiddens,
                               count_embedding_dim= count_embedding_num)
        self.cell_encoder = cell_encoder(d_model, num_hiddens) 

    def forward(self,gene_id, count_id, mask, pred_positions):

        embed = self.embedding_layer(gene_id, count_id) 
        embed = self.sc_encoder(embed, mask= mask)

        cell_embed = embed[:,0,:]
        cell_embed = self.cell_encoder(cell_embed)

        gene_pred = self.mask_gp(embed[:,1:,:],pred_positions[:,1:])  # pred_positions: the first is <cls> token

        return cell_embed, gene_pred