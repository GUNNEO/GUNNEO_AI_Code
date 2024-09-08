from collections import OrderedDict
from typing import Callable, NamedTuple, Optional
from torchvision.ops.misc import Conv3dNormActivation
import torch
import torch.nn as nn


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    # indicates it can accept ant type of input and return nn.Module
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm3d
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


class ViT3D(nn.Module):
    def __init__(
        self,
        image_size: int,
        frame_size: int,
        image_patch: int,
        frame_patch: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_patches: Optional[int] = None,
        conv_stem_config: Optional[ConvStemConfig] = None
    ):
        super().__init__()
        num_patches = (image_size // image_patch) ** 2 * \
            (frame_size // frame_patch) + 1
        torch._assert(image_size % image_patch == 0,
                      "image size should be divisable by image patch")
        torch._assert(frame_size % frame_patch == 0,
                      "frame size should be divisable by frame patch")
        if output_patches is not None:
            torch._assert((num_patches - 1) % output_patches == 0,
                          f"output_patches should be divisable by the number of patches {num_patches - 1}")

        self.image_size = image_size
        self.image_patch = image_patch
        self.frame_size = frame_size
        self.frame_patch = frame_patch
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.output_patches = output_patches

        # initialize the predefined conv network
        if conv_stem_config is not None:
            prev_channels = num_channels
            seq_proj = nn.Sequential()
            for i, conv_stem_layer_config in enumerate(conv_stem_config):
                seq_proj.add_module(f"conv_bn_relu{i}",
                                    Conv3dNormActivation(in_channels=prev_channels,
                                                         out_channels=conv_stem_layer_config.out_channels,
                                                         kernel_size=conv_stem_layer_config.kernel_size,
                                                         stride=conv_stem_layer_config.stride,
                                                         norm_layer=conv_stem_layer_config.norm_layer,
                                                         activation_layer=conv_stem_layer_config.activation_layer))
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module("conv_last", nn.Conv3d(
                in_channels=prev_channels, out_channels=self.hidden_dim, kernel_size=1))
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj: nn.Module = nn.Conv3d(in_channels=self.num_channels, out_channels=self.hidden_dim, kernel_size=(
                self.frame_patch, self.image_patch, self.image_patch), stride=(self.frame_patch, self.image_patch, self.image_patch))

        self.output_patches = num_patches - 1 if output_patches is None else output_patches
        scale = self.hidden_dim ** -0.5
        self.class_token = nn.Parameter(scale * torch.randn(self.hidden_dim))
        self.position_embedding = nn.Parameter(
            scale * torch.randn(num_patches, self.hidden_dim))

        # define encoder
        mlp_dim = hidden_dim * 4
        self.ln_1 = LayerNorm(self.hidden_dim)

        self.encoder = Encoder(num_layers=self.num_layers, num_heads=self.num_heads, hidden_dim=hidden_dim,
                               mlp_dim=mlp_dim, dropout=self.dropout, attention_dropout=self.attention_dropout)

        self.linear = nn.Linear(num_patches - 1, self.output_patches)

        self.ln_2 = LayerNorm(self.hidden_dim)

    def _process_input(self, x: torch.Tensor):
        n, c, d, h, w = x.shape
        img_p = self.image_patch
        frame_p = self.frame_patch
        torch._assert(h == self.image_size, ("Wrong image height! Expected ",
                      str(self.image_size), " but got ", str(h), "!"))
        torch._assert(w == self.image_size, ("Wrong image width! Expected ",
                      str(self.image_size), " but got ", str(w), "!"))
        torch._assert(d == self.frame_size, ("Wrong image depth! Expected ",
                      str(self.frame_size), " but got ", str(d), "!"))

        n_h = h // img_p
        n_w = w // img_p
        n_d = d // frame_p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, (n_h * n_w * n_d))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        x = torch.cat([self.class_token.to(x.dtype) + torch.zeros(x.shape[0],
                      1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.position_embedding.to(x.dtype)
        x = self.ln_1(x)

        # input shape = output shape: (n, num_patches, hidden_dim)
        x = self.encoder(x)
        # centrally focus on the first layer of encoder output, other approches
        # such as linear projection or average pooling can be applied
        # x = self.ln_2(x[:, 0, :])
        cls_token = x[:, 0, :].unsqueeze(dim=1)
        linear_x = self.linear(x[:, 1:, :].permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat((cls_token, linear_x), dim=1)
        x = self.ln_2(x)
        return x


test_tensor = torch.randn(10, 1, 30, 224, 224)
model = ViT3D(image_size=224, frame_size=30, image_patch=16, frame_patch=3, num_channels=1,
              num_heads=4, num_layers=1, hidden_dim=1024, dropout=0.5, attention_dropout=0.5, output_patches=245)
out = model(test_tensor)
print(out.shape)
