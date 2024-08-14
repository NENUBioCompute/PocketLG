import torch
from mol_encoder import AtomEncoder
from gin_conv import GINConv
import torch.nn.functional as F

# GNN to generate node embedding     用于生成节点表示
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module"""

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers  # 图卷积层数表示
        self.drop_ratio = drop_ratio  # dropout比率
        self.JK = JK      # 指示如何将不同层的节点嵌入组合成图嵌入的方法，默认为 "last"。
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim) # 创建了一个原子编码器对象 AtomEncoder，用于将原子特征转换为嵌入表示。

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):  # 接受一个包含节点特征、边索引和边属性的 batched_data 对象 !!!!!!!
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # 将节点特征通过原子编码器转换为节点嵌入向量 h_list
        for layer in range(self.num_layers):
            '''
            调用了存储在 self.convs 中的第 layer 层的图卷积层，
            并传入节点嵌入向量、边索引和边属性作为参数。
            '''
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)  # 对图卷积后的结果 h 应用批标准化层，以减少训练过程中的内部协变量偏移。
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]  # 将最后一层的节点嵌入向量作为整个图的表示
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation  # 返回图级别的表示结果。


