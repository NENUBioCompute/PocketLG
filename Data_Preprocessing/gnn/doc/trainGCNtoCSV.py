import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
import graph_feature
import numpy as np
from kdbnet import model
import csv

# 准备数据集
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_key = list(self.graphs.keys())[idx]
        graph_data = self.graphs[graph_key]  # 获取对应标识符的图数据
        return graph_data

# 加载目标值
def load_targets(file_path):
    targets = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pdb_id, target = line.strip().split()  # 假设目标值与 pdb ID 以空格分隔
            target = target.split('\ufeff')[-1]
            targets[pdb_id] = float(target)
    return targets

# 初始化模型和数据加载器
model = model.Prot3DGraphModel()
pdb_graph_db = graph_feature.MainGraph()
dataset = GraphDataset(pdb_graph_db)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 提取模型的输出特征向量和对应的 target
output_data = []
model.eval()

# 创建CSV文件并写入数据
with open(r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_outGCN.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    with torch.no_grad(), tqdm(total=len(data_loader), desc='Processing data') as pbar:
        for data in data_loader:
            output = model(data)
            # 提取每个 pdb_id，并转换为字符串
            pdb_ids = [str(pdb_id).replace('.pdb', '') for pdb_id in data.name]
            output_list = output.squeeze().tolist()
            # print(pdb_ids,pdb_ids)
            # print(output_list,output_list)
            # 将数据写入CSV文件
            for pdb_id, output_val in zip(pdb_ids, output_list):
                writer.writerow([pdb_id, output_list])
            pbar.update(1)

####   如果GCNtoCSV2很慢，可以用这个提取GCN特征然后手动复制到ESM的csv文件中