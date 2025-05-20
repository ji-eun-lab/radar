# # 📁 data_utils/h5_loader.py
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import Dataset

# class H5PointCloudDataset(Dataset):
#     def __init__(self, h5_path, split='train', test_ratio=0.2, num_points=1024):
#         self.num_points = num_points
#         with h5py.File(h5_path, 'r') as f:
#             self.data = f['data'][:]
#             self.label = f['label'][:]
#         self.label = np.squeeze(self.label)

#         # split
#         total = len(self.label)
#         split_idx = int(total * (1 - test_ratio))
#         if split == 'train':
#             self.data = self.data[:split_idx]
#             self.label = self.label[:split_idx]
#         else:
#             self.data = self.data[split_idx:]
#             self.label = self.label[split_idx:]

#     def __len__(self):
#         return len(self.label)

#     def __getitem__(self, idx):
#         point_set = self.data[idx][:self.num_points]
#         label = self.label[idx]
#         return torch.tensor(point_set, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class H5PointCloudSequenceDataset(Dataset):
    def __init__(self, h5_path, split='train', test_ratio=0.2, num_points=1024, seq_len=5):
        self.num_points = num_points
        self.seq_len = seq_len
        
        with h5py.File(h5_path, 'r') as f:
            self.data = f['data'][:]
            self.label = f['label'][:]
        self.label = np.squeeze(self.label)

        total = len(self.label)
        split_idx = int(total * (1 - test_ratio))
        
        if split == 'train':
            self.data = self.data[:split_idx]
            self.label = self.label[:split_idx]
        else:
            self.data = self.data[split_idx:]
            self.label = self.label[split_idx:]

    def __len__(self):
        return len(self.label) - self.seq_len + 1

    def __getitem__(self, idx):
        seq_data = np.zeros((self.seq_len, 3, self.num_points), dtype=np.float32)
        seq_labels = self.label[idx:idx+self.seq_len]

        for t in range(self.seq_len):
            seq_data[t] = self.data[idx + t][:self.num_points].T  # shape: (3, num_points)

        # 보통 시퀀스의 대표 라벨로 마지막 시점 라벨을 사용
        final_label = seq_labels[-1]

        return torch.tensor(seq_data), torch.tensor(final_label, dtype=torch.long)
