from collections import OrderedDict
from typing import Callable, NamedTuple, Optional, List
from torchvision.ops.misc import Conv2dNormActivation
import torch
import torch.nn as nn
import math
from utils import LayerNorm, BatchNorm2d, QuickGELU


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    # indicates it can accept ant type of input and return nn.Module
    norm_layer: Callable[..., nn.Module] = BatchNorm2d
    activation_layer: Callable[..., nn.Module] = QuickGELU


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
        # batch_first = True, no need to permute the x
        self.attn = nn.MultiheadAttention(
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
            return self.attn_mask.to(dtype=x.dtype, device=x.device)
        else:
            return self.attn_mask

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.attn(
            x, x, x, need_weights=False, attn_mask=self.attention_mask(x))
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attn_mask = attn_mask
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


class ViT(nn.Module):
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
        output_patches: Optional[int] = None,
        conv_stem_configs: Optional[List[ConvStemConfig]] = None
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2 + 1
        torch._assert(image_size % patch_size == 0,
                      f'image size {image_size} indivisable by patch size {patch_size}')
        if conv_stem_configs is not None:
            torch._assert(isinstance(conv_stem_configs, ConvStemConfig),
                          "input conv configuration is not supperted")
        if output_patches is not None:
            torch._assert((num_patches - 1) % output_patches == 0,
                          f"output_patches {output_patches} indivisable by the number of patches {num_patches - 1}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_patches = output_patches
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # define the pre_cov for specific input
        if conv_stem_configs is not None:
            prev_channels = num_channels
            seq_proj = nn.Sequential()
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
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
        scale = hidden_dim ** -0.5
        self.class_token = nn.Parameter(scale * torch.randn(self.hidden_dim))
        self.position_embedding = nn.Parameter(
            scale * torch.randn(num_patches, self.hidden_dim))

        # define encoder
        mlp_dim = hidden_dim * 4
        self.ln_1 = LayerNorm(self.hidden_dim)

        self.encoder = Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout
        )

        if self.output_patches is not None:
            # (n, hidden_dim, num_patches) -> (n, hidden_dim, output_patches)
            self.linear = nn.Linear(num_patches - 1, self.output_patches)

        self.ln_2 = LayerNorm(self.hidden_dim)

        # init the params
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * \
                self.conv_proj.kernel_size[0] * \
                self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight,
                                  std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, ("Wrong image height! Expected ",
                      str(self.image_size), " but got ", str(h), "!"))
        torch._assert(w == self.image_size, ("Wrong image width! Expected ",
                      str(self.image_size), " but got ", str(w), "!"))

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
        x = torch.cat([self.class_token.to(x.dtype) + torch.zeros(x.shape[0],
                      1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.position_embedding.to(x.dtype)
        x = self.ln_1(x)

        # input shape = output shape: (n, num_patches, hidden_dim)
        # the transformer attn expect a input size (seq_length, n, hidden_dim)
        x = self.encoder(x)
        # centrally focus on the first layer of encoder output, other approches
        # such as linear projection or average pooling can be applied
        # x = self.ln_2(x[:, 0, :])
        if self.output_patches is not None:
            cls_token = x[:, 0, :].unsqueeze(dim=1)
            linear_x = self.linear(
                x[:, 1:, :].permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.cat((cls_token, linear_x), dim=1)
        x = self.ln_2(x)
        return x
