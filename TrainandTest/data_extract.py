import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from scipy.io import savemat

# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# MLP模块
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 8))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_dim / 8), output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)  # 新增层
        self.conv4 = nn.Conv1d(128, output_dim, 1)  # 注意：这里改为conv4
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)  # 新增层的批处理归一化
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))  # 新增层的前向传播
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=False)[0]
        return x


# 自定义Dataset类
# class CustomDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         feature_rep_str = self.data.iloc[idx]['Feature Representation']
#         atom_cloud_features_str = self.data.iloc[idx]['CA_coords']
#         gcn_data_str = self.data.iloc[idx]['GCN']  # 用于读取 GCN 特征数据
#
#         # 使用 literal_eval 来去掉字符串中的方括号，并将字符串转换成列表
#         feature_rep = torch.FloatTensor(ast.literal_eval(feature_rep_str))
#         atom_cloud_features = torch.FloatTensor(ast.literal_eval(atom_cloud_features_str))
#         gcn_features = torch.FloatTensor(ast.literal_eval(gcn_data_str))
#
#         # 将 atom_cloud_features 经过 PointNetFeatureExtractor 转换
#         pointnet_extractor = PointNetFeatureExtractor(input_dim=atom_cloud_features.size(1), output_dim=32)
#         atom_cloud_features_transformed = pointnet_extractor(atom_cloud_features.unsqueeze(0)).squeeze(0)
#
#         # 拼接转换后的特征与其他特征
#         combined_features = torch.cat((gcn_features, feature_rep, atom_cloud_features_transformed), dim=0)
#
#         targets = torch.FloatTensor([self.data.iloc[idx]['Targets']])
#
#         return combined_features, targets

# dataset = CustomDataset(r'26138AllFeature.csv')
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_outESM.csv')
#data = pd.read_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data1\esm_features_train5067.csv')

features, labels, names = [], [], []
for idx in range(len(data)):
    feature_rep_str = data.iloc[idx]['ESM']
    gcn_data_str = data.iloc[idx]['GCN']  # 用于读取 GCN 特征数据

    feature_rep = torch.FloatTensor(ast.literal_eval(feature_rep_str))
    gcn_features = torch.FloatTensor(ast.literal_eval(gcn_data_str))


    # 拼接转换后的特征与其他特征

    combined_features = torch.cat((feature_rep, gcn_features), dim=0)

    targets = torch.FloatTensor([data.iloc[idx]['Targets']])
    ProteinID = data.iloc[idx]['Protein ID']  # 保持为字符串类型

    features.append(combined_features.detach().numpy())
    labels.append(targets.detach().numpy())
    names.append(ProteinID)  # 不需要转换为Tensor类型

savemat('protein1xo2ESM2.mat', {
    'features': features,
    'labels': labels,
    'names': names
})
print(1)