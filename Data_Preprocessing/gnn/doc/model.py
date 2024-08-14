# 定义 GCN 模型
import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F

#**********************************Graph Transormer***************************************************
class Prot3DGraphModel(nn.Module):
    def __init__(self,
                 d_vocab=21, d_embed=20,
                 d_dihedrals=6,  d_pretrained_emb=1280, d_edge=39,
                 d_gcn=[128, 256, 256],
                 ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())

        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        print('x!', x.shape)   # 使用 .shape 属性
        s = data.node_s
        print('s!', s.shape)  # 使用 .shape 属性

        x = torch.cat([x, s], dim=-1)
        print('x_catch!', x.shape)  # 使用 .shape 属性

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        x = self.gcn(x, edge_index, edge_attr)
        print('xxx', x.shape)  # 使用 .shape 属性



        x = self.pool(x, batch)  # 全局池化层
        print('x___pool', x.shape)  # 使用 .shape 属性

        #x = self.dropout(x)  # 在全局池化层后应用Dropout

        x = self.fc_output(x)  # 输出层，不使用激活函数
        return x

#**********************************Graph Transormer***************************************************
class Prot3DGraphModel1(nn.Module):
    def __init__(self,
                 d_vocab=21, d_embed=20,
                 d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
                 d_gcn=[128, 256, 256],
                 ):
        super(Prot3DGraphModel1, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn

        layers = []
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())

        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)
        self.pool = torch_geometric.nn.global_mean_pool

        # 添加几个全连接层和激活函数
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, node_s, seq):
        batch = data.batch
        print('node_s', node_s.shape)
        print('seq', seq.shape)
        edge_index = data.edge_index
        # 将 node_s 和 seq 拼接在一起，沿着最后一个维度拼接
        x = torch.cat((node_s, seq), dim=-1)
        # 将第一个维度展平
        x = x.reshape(-1, x.size(2))

        print('x_catch!', x.shape)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)


        x = self.gcn(x, edge_index, edge_attr)
        print('x_gcn', x.shape)  # 使用 .shape 属性
        x = x.view(32, 70, 256)
        print('xxxxx', x.shape)  # 使用 .shape 属性
        pooled_x = torch.mean(x, dim=1)
        print('pooled_x', pooled_x.shape)
        # 添加全连接层和激活函数
        pooled_x = self.relu(self.fc1(pooled_x))
        pooled_x = self.relu(self.fc2(pooled_x))
        pooled_x = self.fc3(pooled_x)
        x = self.sigmoid(pooled_x)  # 将输出转换为概率值

        #print('x:::::', x)

        return x


#**********************************GIN***************************************************
# class Prot3DGraphModel1(nn.Module):
#     def __init__(self,
#                  d_vocab=21, d_embed=20,
#                  d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
#                  d_gcn=[128, 256, 256],
#                  ):
#         super(Prot3DGraphModel1, self).__init__()
#         d_gcn_in = d_gcn[0]
#         self.embed = nn.Embedding(d_vocab, d_embed)
#         self.proj_node = nn.Linear(d_embed + d_dihedrals, d_gcn_in)
#         self.proj_edge = nn.Linear(d_edge, d_gcn_in)
#         gcn_layer_sizes = [d_gcn_in] + d_gcn
#
#         layers = []
#         for i in range(len(gcn_layer_sizes) - 1):
#             layers.append((
#                 torch_geometric.nn.TransformerConv(
#                     gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
#                 'x, edge_index, edge_attr -> x'
#             ))
#             layers.append(nn.LeakyReLU())
#
#         self.gcn = torch_geometric.nn.Sequential(
#             'x, edge_index, edge_attr', layers)
#         self.pool = torch_geometric.nn.global_mean_pool
#
#     def forward(self, data, node_s, seq):
#         edge_index = data.edge_index
#         x = node_s
#         s = seq
#
#         #print('data', data)
#
#         #x = self.embed(x)
#
#         print('x!!!', x.shape)
#         print('s!!!', s.shape)
#         x = torch.cat([x, s], dim=-1)
#         print('拼接后的x和s!!!', x.shape)
#
#         edge_attr = data.edge_s
#
#         x = self.proj_node(x)
#         edge_attr = self.proj_edge(edge_attr)
#
#         #print('edge_attr', edge_attr)
#         # print('edge_attr', edge_attr.shape)
#
#         x = self.gcn(x, edge_index, edge_attr)
#         print('x2', x)
#         print('x2', x.shape)
#
#         num_nodes = 100
#         #x = torch_geometric.nn.global_mean_pool(x, batch)
#         return x



