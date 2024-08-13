import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat


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


# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = loadmat(file_path)
        self.data = data['features']
        self.label = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 拼接转换后的特征与其他特征
        combined_features = self.data[idx][:-32]
        targets = torch.FloatTensor(self.label[idx])
        return combined_features, targets


if __name__ == '__main__':
    # 指定训练数据集路径和测试数据集路径
    train_file = r'merged_file.mat'
    test_file = r'RefinedSetESM_test.mat'

    # 初始化数据集
    train_dataset = CustomDataset(train_file)
    test_dataset = CustomDataset(test_file)

    # 初始化MLP模型、损失函数和优化器
    input_dim = 1536  # PointNet输出特征维度 + GCN特征维度 + 其他特征维度
    hidden_dim = 2048
    output_dim = 1
    criterion = nn.L1Loss()

    # 选择指定的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 初始化MLP模型
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # 初始化数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练模型
    num_epochs = 300
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
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, R2: {train_r2:.4f}, Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'RefinedSetESM_model4.pth')
