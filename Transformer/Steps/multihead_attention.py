import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sequence_length = 4  # 4 words
batch_size = 1  # 1 sentence per input
input_dim = 1024
d_model = 512  # 512 dimensions
x = torch.randn((batch_size, sequence_length, input_dim))

qkv_layer = nn.Linear(input_dim, 3 * d_model)  # map query, key and value
qkv = qkv_layer(x)

num_heads = 8
head_dim = d_model // num_heads  # split one head into 8 parts
qkv = qkv.reshape(batch_size, num_heads, sequence_length, 3 * head_dim)
q, k, v = qkv.chunk(3, dim=-1)

# attention part
d_k = q.size()[-1]
# obtain attention score
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)  # generate the lower triangle
scaled += mask
attention = F.softmax(scaled, dim=-1)
values = torch.matmul(attention, v)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:  # to determine whether it is encoder or decoder
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads,
                          3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.d_model)
        out = self.linear_layer(values)
        return out


input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
x = torch.randn((batch_size, sequence_length, input_dim))
model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)
