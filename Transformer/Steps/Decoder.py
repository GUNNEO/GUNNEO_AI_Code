import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product(q, k, v, mask=None):
    '''q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64,
    v: 30 x 8 x 200 x 64, mask 200 x 200'''
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / \
        math.sqrt(d_k)  # 30 x 8 x 200 x 200
    if mask is not None:
        scaled += mask  # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1)  # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v)  # 30 x 8 x 200 x 64
    return values, attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # 512
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # 512

    def forward(self, inputs):
        # inputs : 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  # [-1]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)  # 1536
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length,
                          self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.d_model)
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)  # 1024
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()  # 30 x 200 x 512
        # transfer the info from encoder
        kv = self.kv_layer(x)  # 30 x 200 x 1024
        q = self.q_layer(y)  # 30 x 200 x 512
        kv = kv.reshape(batch_size, sequence_length, self.num_heads,
                        2 * self.head_dim)  # 30 x 200 x 8 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads,
                      self.head_dim)  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 64
        # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(
            q, k, v, mask)  # 30 x 8 x 200 x 64
        values = values.reshape(
            batch_size, sequence_length, d_model)  # 30 x 200 x 512
        out = self.linear_layer(values)  # 30 x 200 x 512
        return out  # 30 x 200 x 512


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        # x for encoder and y for decoder, only y would change
        _y = y  # 30 x 200 x 512
        y = self.self_attention(y, mask=decoder_mask)  # 30 x 200 x 512
        y = self.dropout1(y)  # 30 x 200 x 512
        y = self.norm1(y + _y)  # 30 x 200 x 512

        _y = y  # 30 x 200 x 512
        y = self.encoder_decoder_attention(x, y, mask=None)  # 30 x 200 x 512
        y = self.dropout2(y)
        y = self.norm2(y + _y)  # 30 x 200 x 512

        _y = y  # 30 x 200 x 512
        y = self.ffn(y)  # 30 x 200 x 512
        y = self.dropout3(y)  # 30 x 200 x 512
        y = self.norm3(y + _y)  # 30 x 200 x 512
        return y  # 30 x 200 x 512


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden,
                 num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(
            d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)])

    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y


d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

# English sentence positional encoded
x = torch.randn((batch_size, max_sequence_length, d_model))
# Kannada sentence positional encoded
y = torch.randn((batch_size, max_sequence_length, d_model))
mask = torch.full([max_sequence_length, max_sequence_length], float('-inf'))
mask = torch.triu(mask, diagonal=1)
decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = decoder(x, y, mask)
