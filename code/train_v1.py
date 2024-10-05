import torch 
import os

def train_one_epoch(model,train_loader,criterion,optimizer,device,gradient_accumulation_steps, save_steps, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练模型
    model.train()
    model.to(device)


    # 训练模型
    total_steps = 0  # 记录总的训练步数
    accumulated_loss = 0.0  # 累积损失
    accumulated_exp_loss = 0.0 
    accumulated_clip_loss = 0.0
    accumulated_genes = 0  # 累积样本数
    accumulated_cells = 0 

    for i, batch in enumerate(train_loader, 0):
        (counts, token_id, mask_array, label) = batch 

        counts = counts.to(device)
        token_id = token_id.to(device)
        mask_array = mask_array.to(device)
        label = label.to(device)

        # 前向传播
        cell_embed, gene_pred = model(token_id, counts, mask_array, mask_array)
        N_c = int(cell_embed.shape[0] / 2)
        N_g = gene_pred.shape[0]

        cell_0 = cell_embed[:N_c,:]
        cell_1 = cell_embed[N_c:,:]
        loss = criterion(cell_0,cell_1, gene_pred, label)
        
        total_loss = loss['total_loss']
        exp_loss = loss['gene_exp_ce_loss']
        clip_loss = loss['clip_loss']
        

        # 计算累积损失和样本数
        
        accumulated_exp_loss += exp_loss.item()*N_g
        accumulated_clip_loss += clip_loss.item()*N_c
        accumulated_loss += accumulated_clip_loss + accumulated_exp_loss
        
        accumulated_cells += N_c 
        accumulated_genes += N_g 
        
        
        # 反向传播
        total_loss.backward()

        # 梯度累积
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度
            total_steps += 1

            # 打印累积损失
            
            avg_exp_loss = accumulated_exp_loss/ accumulated_genes
            avg_clip_loss = accumulated_clip_loss / accumulated_cells  
            avg_loss = avg_exp_loss + avg_clip_loss 
            print(f'Step {total_steps}, Loss: {avg_loss:.4f}, Exp_loss : {avg_exp_loss:.4f}, Clip_loss ; {avg_clip_loss:.4f}')

            # 重置累积损失和样本数
            accumulated_loss = 0.0
            accumulated_exp_loss = 0.0 
            accumulated_clip_loss = 0.0
            accumulated_cells = 0
            accumulated_genes = 0

            # 模型保存
            if total_steps % save_steps == 0:
                checkpoint = {
                    'steps': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{total_steps}.pth'))
                print(f"Checkpoint saved at steps {total_steps}")

        if (i + 1) == len(train_loader): # last batch update model
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度
            total_steps += 1

            # 打印累积损失
            avg_exp_loss = accumulated_exp_loss/ accumulated_genes
            avg_clip_loss = accumulated_clip_loss / accumulated_cells  
            avg_loss = avg_exp_loss + avg_clip_loss 
            print(f'Step {total_steps}, Loss: {avg_loss:.4f}, Exp_loss : {avg_exp_loss:.4f}, Clip_loss ; {avg_clip_loss:.4f}')

            # 重置累积损失和样本数
            accumulated_loss = 0.0
            accumulated_exp_loss = 0.0 
            accumulated_clip_loss = 0.0
            accumulated_cells = 0
            accumulated_genes = 0

            # 模型保存
            if total_steps % save_steps == 0:
                checkpoint = {
                    'steps': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{total_steps}.pth'))
                print(f"Checkpoint saved at steps {total_steps}")


            print('Finished Training')

    return model


def eval_one_epoch(model,eval_loader,criterion,device, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    model.to(device)


    # 训练模型
    total_loss = 0 
    exp_loss = 0
    clip_loss = 0
    total_N_c = 0 
    total_N_g = 0 
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader, 0):

            (counts, token_id, mask_array, label) = batch 

            counts = counts.to(device)
            token_id = token_id.to(device)
            mask_array = mask_array.to(device)
            label = label.to(device)

            cell_embed, gene_pred = model(token_id, counts, mask_array, mask_array)

            N_c = int(cell_embed.shape[0] / 2)
            N_g = gene_pred.shape[0] 

            cell_0 = cell_embed[:N_c,:]
            cell_1 = cell_embed[N_c:,:]
            loss = criterion(cell_0,cell_1, gene_pred, label)
            
            #total_loss += loss['total_loss'].item() * N * 2

            exp_loss += loss['gene_exp_ce_loss'].item() * N_g
            clip_loss += loss['clip_loss'].item() * N_c
            total_N_c += N_c 
            total_N_g += N_g 

    #avg_total_loss = total_loss / len(eval_loader)
    #avg_exp_loss = exp_loss / len(eval_loader)
    #avg_clip_loss = clip_loss / len(eval_loader) 

    avg_exp_loss = exp_loss / total_N_g 
    avg_clip_loss = clip_loss / total_N_c
    avg_total_loss = avg_exp_loss + avg_clip_loss 
    
    print(f'avg total loss:{avg_total_loss:.4f}, avg exp loss:{avg_exp_loss:.4f}, avg_clip_loss:{avg_clip_loss:.4f}')

    return (avg_total_loss, avg_exp_loss, avg_clip_loss)

    
def train_multi_epoch(model,
                      train_loader,
                      eval_loader, 
                      criterion,
                      optimizer,
                      device,
                      gradient_accumulation_steps, 
                      save_steps, 
                      save_dir, 
                      epochs):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(epochs):
        print(f'epochs: {i} begin:' + '===='*20)
        
        epoch_dir = os.path.join(save_dir,f'epoch_{i}')
        model = train_one_epoch(model,
                                train_loader,
                                criterion,
                                optimizer,
                                device,
                                gradient_accumulation_steps, 
                                save_steps, 
                                save_dir = epoch_dir)
        
        print(f'model evaluation' + '===='*20)
        
        (avg_total_loss, avg_exp_loss, avg_clip_loss) = eval_one_epoch(model,
                                                                       eval_loader,
                                                                       criterion,
                                                                       device, 
                                                                       save_dir = epoch_dir)
        with open(os.path.join(save_dir, 'eval_result'), 'w') as f:
            f.write(f'epoch {i} eval results:\n')
            f.write(f'avg_total_loss:{avg_total_loss}\n')
            f.write(f'avg_exp_loss:{avg_exp_loss}\n')
            f.write(f'avg_clip_loss:{avg_clip_loss}\n')
        f.close()  
            
    return model
