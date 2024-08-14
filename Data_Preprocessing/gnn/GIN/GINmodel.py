# 定义 GCN 模型
import torch
import torch_geometric
from torch import nn
#**********************************GIN***************************************************
class Prot3DGraphModel(nn.Module):
    def __init__(self,
                 d_vocab=21, d_embed=20,
                 d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
                 d_gcn=[128, 256, 256],
                 ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn

        0
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
        print('data', data)

        x = self.embed(x)
        s = data.node_s
        x = torch.cat([x, s], dim=-1)
        print('x', x)
        print('x', x.shape)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        print('edge_attr', edge_attr)
        print('edge_attr', edge_attr.shape)

        x = self.gcn(x, edge_index, edge_attr)
        print('x2', x)
        print('x2', x.shape)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x



