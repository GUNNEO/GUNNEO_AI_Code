import math
import torch
import torch.nn as nn
from typing import Optional, List


def mask(
    seq_length: int
):
    mask = torch.full(
        [seq_length, seq_length], float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class DualSequential(nn.Module):
    def __init__(
        self,
        layers: List
    ):
        super(DualSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        for layer in self.layers:
            x, y = layer(x, y)
        return x, y


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class positionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        dropout: float
    ):
        super().__init__()
        pe = torch.zeros(seq_length, d_model)
        position = (torch.arange(start=0, end=seq_length).to(
            torch.float32)).unsqueeze(1)
        # e^(2i * (-ln(10000) / d_model))
        div_term = torch.exp(
            (torch.arange(start=0, end=d_model, step=2).to(torch.float32)) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model), 1 used for broadcast
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor
    ):
        # let the model not so sensetive to the absolute position
        return self.dropout(x + self.pe)


class multiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = (d_model // num_heads) ** (-0.5)
        self.qkv_layer = nn.Linear(d_model, d_model * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        b, l, d = x.shape
        res = x
        x = self.qkv_layer(x)
        x = x.reshape(b, l, self.num_heads,
                      (self.d_model // self.num_heads) * 3)
        x = x.permute(0, 2, 1, 3)
        q, k, v = x.chunk(3, dim=-1)
        kt = torch.transpose(k, dim0=-2, dim1=-1)
        # Attention(Q, K, V) with multiple heads
        if mask is None:
            x = torch.matmul(self.softmax(torch.matmul(q, kt) * self.scale), v)
        else:
            x = torch.matmul(
                (self.softmax(torch.matmul(q, kt) * self.scale) + mask), v)
        x = x.reshape(b, l, self.d_model)
        x = self.dropout(x)
        x = x + res
        x = self.ln(x)
        return x


#  used for decoder layer
class multiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = (d_model // num_heads) ** (-0.5)
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, d_model * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(d_model)

    # x: encoder memory
    # y: decoder query
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        b, l, d = x.shape
        res = y
        y = self.q_layer(y)
        x = self.kv_layer(x)
        y = y.reshape(b, l, self.num_heads,
                      self.d_model // self.num_heads)
        x = x.reshape(b, l, self.num_heads,
                      (self.d_model // self.num_heads) * 2)
        q = y.permute(0, 2, 1, 3)
        x = x.permute(0, 2, 1, 3)
        k, v = x.chunk(2, dim=-1)
        kt = torch.transpose(k, dim0=-2, dim1=-1)
        # Attention(Q, K, V) with multiple heads
        y = torch.matmul(self.softmax(torch.matmul(q, kt) * self.scale), v)
        y = y.reshape(b, l, self.d_model)
        y = self.dropout(y)
        y = y + res
        y = self.ln(y)
        return y


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.gelu = QuickGELU()
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor
    ):
        res = x
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + res
        return x


class encoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        seq_length: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.multiAttn = multiHeadAttention(
            d_model=d_model,
            seq_length=seq_length,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = FFN(
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.multiAttn(x)
        x = self.ffn(x)
        return x


class decoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        seq_length: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.maskedMultiAttn = multiHeadAttention(
            d_model=d_model,
            seq_length=seq_length,
            num_heads=num_heads,
            dropout=dropout
        )
        self.crossAttn = multiHeadCrossAttention(
            d_model=d_model,
            seq_length=seq_length,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = FFN(
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout
        )
        self.mask = mask(seq_length=seq_length)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        y = self.maskedMultiAttn(y, mask=self.mask)
        y = self.crossAttn(x, y)
        y = self.ffn(y)
        return x, y


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size1: int,
        vocab_size2: int,
        d_model: int,
        d_inner: int,
        seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.embedding1 = nn.Embedding(
            num_embeddings=vocab_size1, embedding_dim=d_model)
        self.embedding2 = nn.Embedding(
            num_embeddings=vocab_size2, embedding_dim=d_model)
        self.pe = positionalEncoding(
            d_model=d_model,
            seq_length=seq_length,
            dropout=dropout
        )
        # define encoder
        encoder_layers = [
            encoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                seq_length=seq_length,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        # define decoder
        decoder_layers = [
            decoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                seq_length=seq_length,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.decoder = DualSequential(decoder_layers)
        # final ouput for probabilities
        self.linear = nn.Linear(d_model, vocab_size2)
        self.linear.weight = self.embedding2.weight
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        # omit the embedding part
        x = self.pe(x)
        y = self.pe(y)
        x = self.encoder(x)
        y = self.decoder(x, y)[1]
        y = self.linear(y)
        output_p = self.softmax(y)
        return output_p


if __name__ == "__main__":
    x = torch.randn(2, 256, 512)
    y = torch.randn(2, 256, 512)
    model = Transformer(200000, 150000, 512, 2048, 256, 8, 6, 0.1)
    out = model(x, y)
    print(out.shape)
