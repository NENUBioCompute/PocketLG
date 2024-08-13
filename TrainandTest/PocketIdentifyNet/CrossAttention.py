from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        #attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """

    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size  # 输入维度
        self.all_head_size = all_head_size  # 输出维度
        self.num_heads = head_num  # 注意头的数量
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def forward(self, features):
        #print('xxxx', features.shape)
        # x = features[:, :-1280]
        # y = features[:, 256:]
        x = features
        y = features
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        y = y.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # print('xxxx',x.shape)
        # print('yyyy',y.shape)
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """
        #linear_q = nn.Linear(256, 1280)  # 由于 y 的维度为 1280，因此线性变换的输出维度也应为 1280
        #x = linear_q(x)  # q 的形状为 [batch_size, seq_length, 1280]
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        #attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s, k_s, v_s)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)
        output = output.squeeze(1)
        #print('output', output.shape)

        input_dim = 1280  # PointNet输出特征维度 + GCN特征维度 + 其他特征维度
        hidden_dim = 2048
        output_dim = 1

        mlp = SimpleMLP(input_dim, hidden_dim, output_dim)
        outputs = mlp(output)
        #print('outputs mlp', outputs)
        return outputs



# MLP模块
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0)
        self.fc3 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_dim / 8), output_dim)

    def forward(self, x):
        # x = self.transformer(x)
        #print('X@@@@@@@@@@@@@@@@',x.shape)
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