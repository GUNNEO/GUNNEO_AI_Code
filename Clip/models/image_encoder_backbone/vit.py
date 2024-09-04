from collections import OrderedDict
from typing import Callable, NamedTuple, Optional, List
from torchvision.ops.misc import Conv2dNormActivation
import torch
import torch.nn as nn


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    # indicates it can accept ant type of input and return nn.Module
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class LayerNorm(nn.LayerNorm):
    '''Temporarily convert precision to fp32'''

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # inherit the forward method from layernorm in torch
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_dim, mlp_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_dim, hidden_dim))
        ]))
        self.attn_mask = attn_mask

    def attention_mask(self, x: torch.Tensor):
        if self.attn_mask is not None:
            return self.attn_mask.to(dtype=x.type, device=x.type)
        else:
            return x

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(
            x, x, x, need_weights=False, attn_mask=self.attention_mask(self.attn_mask))
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    '''make encoder latent layers for vit'''

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        attn_mask: torch.Tensor = None
    ):
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                attn_mask
            )
        self.layers = nn.Sequential(layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_dim: Optional[int] = None,
        representation_size: Optional[int] = None,
        # ConvStemConfig can be ConvStemConfig type or None
        conv_stem_config: Optional[List[ConvStemConfig]] = None
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0,
                      'image shape indivisable by patch size')
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # define the pre_cov for specific input
        if conv_stem_config is not None:
            prev_channels = num_channels
            seq_proj = nn.Sequential()
            for i, conv_stem_layer_config in enumerate(conv_stem_config):
                seq_proj.add_module(f"conv_bn_relu{i}",
                                    Conv2dNormActivation(in_channels=prev_channels,
                                                         out_channels=conv_stem_layer_config.out_channels,
                                                         kernel_size=conv_stem_layer_config.kernel_size,
                                                         stride=conv_stem_layer_config.stride,
                                                         norm_layer=conv_stem_layer_config.norm_layer,
                                                         activation_layer=conv_stem_layer_config.activation_layer))
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module("conv_last", nn.Conv2d(
                in_channels=prev_channels, out_channels=self.hidden_dim, kernel_size=1))
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj: nn.Module = nn.Conv2d(
                in_channels=num_channels, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # add class tokens and position embedding
        num_patches = (image_size // patch_size) ** 2 + 1
        scale = hidden_dim ** -0.5
        self.class_token = nn.Parameter(scale * torch.randn(self.hidden_dim))
        self.position_embedding = nn.Parameter(
            scale * torch.randn(num_patches, self.hidden_dim))

        # define encoder
        mlp_dim = hidden_dim * 4
        self.output_dim = output_dim
        self.ln_1 = LayerNorm(self.hidden_dim)

        self.encoder = Encoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=hidden_dim,
                               mlp_dim=mlp_dim, dropout=self.dropout, attention_dropout=self.attention_dropout)

        self.ln_2 = LayerNorm(self.hidden_dim)
        if self.output_dim is not None:
            self.final_porj_weights = nn.Parameter(
                scale * torch.randn(self.hidden_dim, self.output_dim))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height! Expected " +
                      str(self.image_size) + " but got " + str(h) + "!")
        torch._assert(w == self.image_size, "Wrong image width! Expected " +
                      str(self.image_size) + " but got " + str(w) + "!")

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)  # shape: (n, (n_h, n_w), hidden_dim)
        x = torch.cat([self.class_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      # shape = (n, (image_size // patch_size ** 2 + 1), hidden_dim)
                                                                  dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.position_embedding.to(x.dtype)
        x = self.ln_1(x)

        # input shape = output shape: (n, (image_size // patch_size ** 2 + 1), hidden_dim)
        x = self.encoder(x)
        x = self.ln_2(x)
        return x


test_tensor = torch.rand(30, 30, 224, 224)  # (n, c, h, w)
vit = VisionTransformer(image_size=224, patch_size=16,
                        num_channels=30, num_layers=2, num_heads=8, hidden_dim=768)
out = vit(test_tensor)
print(out.shape)
