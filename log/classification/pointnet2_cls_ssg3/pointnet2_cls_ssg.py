# import torch.nn as nn
# import torch.nn.functional as F
# from pointnet2_utils import PointNetSetAbstraction


# class get_model(nn.Module):
#     def __init__(self,num_class,normal_channel=True):
#         super(get_model, self).__init__()
#         in_channel = 6 if normal_channel else 3
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)


#         return x, l3_points



# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

class PointNetLSTM(nn.Module):
    def __init__(self, num_class, normal_channel=True, lstm_hidden_size=256, lstm_layers=1):
        super(PointNetLSTM, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # 기존 PointNet
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)

        # LSTM layer 추가
        self.lstm = nn.LSTM(input_size=1024, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # 분류를 위한 Fully Connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz_seq):
        # xyz_seq: (batch, seq_len, channels, points)
        B, seq_len, C, N = xyz_seq.shape
        outputs = []

        for t in range(seq_len):
            xyz = xyz_seq[:, t, :, :]  # 시계열의 각 timestep 처리
            if self.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                norm = None

            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            _, l3_points = self.sa3(l2_xyz, l2_points)

            outputs.append(l3_points.view(B, -1))

        # LSTM 입력 형태로 변환 (batch, seq_len, features)
        lstm_input = torch.stack(outputs, dim=1)  # (B, seq_len, 1024)

        # LSTM 처리
        lstm_out, _ = self.lstm(lstm_input)  # lstm_out: (B, seq_len, lstm_hidden_size)

        # 마지막 시점의 출력 사용
        lstm_final_output = lstm_out[:, -1, :]  # (B, lstm_hidden_size)

        # Fully connected layers
        x = self.drop1(F.relu(self.bn1(self.fc1(lstm_final_output))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)

        return x, lstm_final_output

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
