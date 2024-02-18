import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat(x, n_rep):
    batch_size, seq_len, qk_num_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(batch_size, seq_len, qk_num_heads, n_rep, head_dim).reshape(batch_size, seq_len, qk_num_heads*n_rep, head_dim)

class GroupedQueryAttention(nn.Module):
    def __init__(self, input_dim, num_heads, n_rep):
        super().__init__()
        self.d_model = input_dim
        self.n_rep = n_rep
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.Wq = nn.Linear(input_dim, self.d_model)
        self.Wk = nn.Linear(input_dim, self.d_model//n_rep)
        self.Wv = nn.Linear(input_dim, self.d_model//n_rep)
        self.Wo = nn.Linear(input_dim, self.d_model)
    
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        q = self.Wq(x).view(batch_size, seq_len, num_heads, self.head_dim)
        k = self.Wk(x).view(batch_size, seq_len, num_heads//self.n_rep, self.head_dim)
        v = self.Wv(x).view(batch_size, seq_len, num_heads//self.n_rep, self.head_dim)

        k = repeat(k, self.n_rep)
        v = repeat(v, self.n_rep)

        attn = F.softmax(torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.d_model ** 0.5))
        output = self.Wo(torch.einsum('bhqk,bkhd->bqhd', attn, v).reshape(batch_size, seq_len, input_dim))
        return output

batch_size = 4
input_dim = 512
num_heads = 8
n_rep = 4
seq_len = 10

x = torch.randn(batch_size, seq_len, input_dim)

GQA = GroupedQueryAttention(input_dim, num_heads, n_rep)   
print(f'output dim: {GQA(x).shape}')     
        

