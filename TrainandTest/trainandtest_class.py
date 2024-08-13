import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat

# MLP模块
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_dim / 8), 1)

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


# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = loadmat(csv_file)
        self.data = data['features']
        self.label = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        combined_features = self.data[idx][:-32]  # 假设只使用前面的特征
        # 将标签转换为二分类, 把大于0.9的认为是正样本，小于0.9的认为是负样本
        label = 1 if self.label[idx] > 0.9 else 0
        return combined_features, label


if __name__ == '__main__':
    # 初始化数据集
    train_dataset = CustomDataset(r'trainESM2.mat')
    test_dataset = CustomDataset(r'testESM2.mat')
    yuzhi = 0.75
    # 初始化MLP模型、损失函数和优化器
    input_dim = 1536  # PointNet输出特征维度 + GCN特征维度 + 其他特征维度
    hidden_dim = 2048
    criterion = nn.BCEWithLogitsLoss()

    # 选择指定的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用所有可见的 GPU
    print("Using device:", device)

    # 初始化MLP模型
    model = SimpleMLP(input_dim, hidden_dim).to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # 初始化数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练模型
    num_epochs = 300
    best_epoch = -1
    best_acc = -1
    best_mcc = -1
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
            loss = criterion(outputs, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
            predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(targets_list, [1 if p >= 0.9 else 0 for p in predictions])
        train_mcc = matthews_corrcoef(targets_list, [1 if p >= 0.9 else 0 for p in predictions])

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
                loss = criterion(outputs, targets.unsqueeze(1).float())
                test_loss += loss.item() * len(inputs)
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
        test_loss /= len(test_dataset)

        # 计算评价指标
        acc = accuracy_score(targets_list, [1 if p >= yuzhi else 0 for p in predictions])
        mcc = matthews_corrcoef(targets_list, [1 if p >= yuzhi else 0 for p in predictions])

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, MCC: {train_mcc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Acc: {acc:.4f}, MCC: {mcc:.4f}")

        # Update best performance
        if acc > best_acc:
            best_epoch = epoch + 1
            best_acc = acc
            best_mcc = mcc

    # Print the results of the best performing epoch
    print(f"Best performing epoch: {best_epoch}, Acc: {best_acc:.4f}, MCC: {best_mcc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f'modelTMalign.pth')

    print(f'Accuracy: {acc}  MCC: {mcc}')
