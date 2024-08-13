import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
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
        self.names = data['names'].flatten()  # 假设names是1D数组

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        combined_features = self.data[idx][:1536]
        targets = torch.FloatTensor(self.label[idx])
        name = self.names[idx]
        return name, combined_features, targets

if __name__ == '__main__':
    # 指定测试数据集路径
    test_file = r'../protein1xo2ESM.mat'  # Test100ESM.mat只有1280+256维

    # 初始化数据集
    test_dataset = CustomDataset(test_file)

    # 初始化MLP模型、损失函数
    input_dim = 1536  # PointNet输出特征维度 + GCN特征维度 + 其他特征维度
    hidden_dim = 2048
    output_dim = 1
    criterion = nn.L1Loss()

    # 选择指定的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    # 初始化MLP模型并加载预训练模型权重
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load('../merge_matv2_fold4.pth'))

    # 初始化数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 测试模型
    model.eval()
    test_loss = 0.0
    predictions = []
    targets_list = []
    names_list = []
    with torch.no_grad():
        for names, inputs, targets in test_dataloader:
            # 将数据移动到 GPU
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * len(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            names_list.extend(names)

    test_loss /= len(test_dataset)

    # # 计算评价指标
    # mse = mean_squared_error(targets_list, predictions)
    # mae = mean_absolute_error(targets_list, predictions)
    # r2 = r2_score(targets_list, predictions)
    #
    # print(f"测试损失: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 将预测结果、真实值和样本名字保存到 CSV 文件
    results = pd.DataFrame({
        'Name': names_list,
        'Target': np.array(targets_list).flatten(),
        'Prediction': np.array(predictions).flatten()
    })
    results.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to test_predictions.csv")
