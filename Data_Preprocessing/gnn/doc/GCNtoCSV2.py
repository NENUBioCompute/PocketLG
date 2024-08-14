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
with torch.no_grad():
    for data in tqdm(data_loader, desc='Processing data'):
        output = model(data)
        # 提取每个 pdb_id，并转换为字符串
        pdb_ids = [str(pdb_id).replace('.pdb', '') for pdb_id in data.name]
        output_list = output.squeeze().tolist()
        # print(pdb_ids)
        # print(output_list)

        # 读取CSV文件中的Protein ID列
        protein_ids = []
        csv_file = r'E:\PDBBind\Result\refinedset_tmscore_70ac\RefinedSetESM.csv'
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                protein_ids.append(row[0])

        # 匹配pdb_id并更新'GCN'列
        gcn_index = protein_ids.index('GCN') if 'GCN' in protein_ids else None
        if gcn_index is not None:
            for pdb_id in pdb_ids:
                if pdb_id in protein_ids:
                    row_idx = protein_ids.index(pdb_id)
                    # 更新'GCN'列中对应行的值
                    protein_ids[row_idx] = output_list

            # 将更新后的'GCN'列写回CSV文件
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(protein_ids)
