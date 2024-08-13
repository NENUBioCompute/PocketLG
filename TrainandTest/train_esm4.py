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
from scipy.io import loadmat

# 设置可见的GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


# MLP模块
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_dim / 8), output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def normalized(data, axis=0, method='maxmin'):
    if method == 'meanstd':
        x_mean = np.mean(data, axis=axis, keepdims=True)  # 保持数据维度
        x_var = np.var(data, axis=axis, keepdims=True)
        return (data - x_mean) / (np.sqrt(x_var) + 1e-8)
    elif method == 'maxmin':
        x_max = np.max(data, axis=axis, keepdims=True)
        x_min = np.min(data, axis=axis, keepdims=True)
        return (data - x_min) / ((x_max - x_min) + 1e-8)

# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = loadmat(csv_file)
        self.data = data['features']
        self.label = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        combined_features = self.data[idx][:-32]
        targets = torch.FloatTensor(self.label[idx])

        return combined_features, targets

if __name__ == '__main__':
    # 初始化数据集
    dataset = CustomDataset(r'merged_file.mat')

    # 初始化MLP模型、损失函数和优化器
    input_dim = 1536  # PointNet输出特征维度 + GCN特征维度 + 其他特征维度
    hidden_dim = 2048
    output_dim = 1
    criterion = nn.L1Loss()

    # 选择指定的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用所有可见的 GPU
    print("Using device:", device)

    # 定义五折交叉验证
    kf = KFold(n_splits=5, shuffle=True)
    val_r2, val_mse = [], []
    for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
        # 划分数据集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # 初始化MLP模型
        model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

        # 定义优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        # 初始化数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 训练模型
        num_epochs = 200
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            predictions = []
            targets_list = []
            for inputs, targets in train_dataloader:
                # 将数据移动到 GPU
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(inputs)
                predictions.extend(outputs.detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())
            train_loss /= len(train_dataset)
            train_r2 = r2_score(targets_list, predictions)

            # 测试模型
            model.eval()
            test_loss = 0.0
            predictions = []
            targets_list = []
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    # 将数据移动到 GPU
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * len(inputs)
                    predictions.extend(outputs.cpu().numpy())
                    targets_list.extend(targets.cpu().numpy())
            test_loss /= len(test_dataset)

            # 计算评价指标
            mse = mean_squared_error(targets_list, predictions)
            mae = mean_absolute_error(targets_list, predictions)
            r2 = r2_score(targets_list, predictions)

            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f'merge_mat_fold{fold + 1}.pth')

        val_r2.append(r2)
        val_mse.append(mse)

    print(f'R2: {np.mean(val_r2)}  MSE: {np.mean(mse)}')
