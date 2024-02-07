import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiQueryAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim//num_heads)
        self.value_linear = nn.Linear(input_dim, hidden_dim//num_heads)
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)
        
        # 线性变换
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # 将输入分割为多个头
        query = query.view(batch_size, seq_length, self.num_heads, self.hidden_dim // self.num_heads)
        key = key.unsqueeze(2)  # 添加一个头的维度
        value = value.unsqueeze(2)  # 添加一个头的维度
        
        # 使用einsum计算注意力权重
        scores = torch.einsum('bqhd,bkhd->bhqk', query, key)
        scores = scores / (self.hidden_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # 扩展维度以适应scores的形状
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # 使用einsum计算加权和
        attended_values = torch.einsum('bhqk,bkhd->bqhd', attention_weights, value)
        
        # 将多个头的结果连接在一起
        attended_values = attended_values.contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # 线性变换并返回结果
        output = self.output_linear(attended_values)
        return output
    
def main():
    # 定义输入张量
    batch_size = 4
    seq_length = 10
    input_dim = 32
    hidden_dim = 64
    num_heads = 8
    
    query = torch.randn(batch_size, seq_length, input_dim)
    key = torch.randn(batch_size, seq_length, input_dim)
    value = torch.randn(batch_size, seq_length, input_dim)
    
    # 创建MultiQueryAttention实例
    mha = MultiQueryAttention(input_dim, hidden_dim, num_heads)
    
    # 前向传播
    output = mha(query, key, value)
    
    # 打印输出形状
    print("Output shape:", output.shape)

if __name__ == '__main__':
    main()