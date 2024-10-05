from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        # 返回所有数据集的总长度
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        # 找出哪个数据集包含索引 idx，并返回相应的样本
        total_len = 0
        for i, dataset in enumerate(self.datasets):
            if idx < len(dataset) + total_len:
                return self.datasets[i][idx - total_len]
            total_len += len(dataset)
        
        raise IndexError("Index out of range") 