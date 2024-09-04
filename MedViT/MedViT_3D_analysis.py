# the model which can take 3D MRI image as input
from functools import partial
import math
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from MedViT.utils_MedViT import merge_pre_bn

NORM_EPS = 1e-5


class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1,
                              groups=groups,
                              bias=False)
        self.norm = nn.BatchNorm3d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        print("ConvBNReLU:")
        x = self.conv(x)
        print("after Conv3d: ", x.shape)
        x = self.norm(x)
        x = self.act(x)
        print("after norm and relu: ", x.shape)
        print("----------")
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool3d(
                (2, 2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        print("PatchEmbed:")
        out = self.norm(self.conv(self.avgpool(x)))
        print("after PatchEmbed: ", out.shape)
        print("----------")
        return out


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3x3 = nn.Conv3d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=out_channels // head_dim,
                                         bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv3d(
            out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        print("MHCA:")
        out = self.group_conv3x3x3(x)
        print("after group_conv3x3x3: ", out.shape)
        out = self.norm(out)
        out = self.act(out)
        print("after norm and relu:", out.shape)
        out = self.projection(out)
        print("after projection with Conv3d:", out.shape)
        print("----------")
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        # insure the kernel size is a odd number
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # change HxWxD -> 1x1x1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        print("ECALayer:")
        y = self.avg_pool(x)
        print("after avgpool: ", y.shape)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        print("after squeeze and transpose: ", y.shape)
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        print("after reshape: ", y.shape)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        print("after dot: ", out.shape)
        print("----------")
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        print("SELayer:")
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print("after avgpool: ", y.shape)
        y = self.fc(y).view(b, c, 1, 1, 1)
        print("after fc: ", y.shape)
        # BxCx1x1x1 -> BxCxHxWxD
        out = x * y
        print("after dot: ", out.shape)
        print("----------")
        return out


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 stride,
                 expand_ratio=4.,
                 act='hs+se',
                 reduction=4,
                 wo_dp_conv=False,
                 dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv3d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride,
                          kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError(
                    'Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1x1 convolution.
        layers.extend([
            nn.Conv3d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        print("LFFN:")
        x = x + self.conv(x)
        print("after LFFN: ", x.shape)
        print("----------")
        return x


class Mlp(nn.Module):
    def __init__(self, in_features,
                 out_features=None,
                 mlp_ratio=None,
                 drop=0.,
                 bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv3d(in_features, hidden_dim,
                               kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_dim, out_features,
                               kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        print("MLP:")
        x = self.conv1(x)
        print("after 1st Conv3d: ", x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        print("after 2nd Conv3d: ", x.shape)
        x = self.drop(x)
        print("----------")
        return x


class ECB(nn.Module):
    """
    Efficient Convolution Block
    """

    def __init__(self, in_channels,
                 out_channels,
                 stride=1,
                 path_dropout=0,
                 drop=0,
                 head_dim=32,
                 mlp_ratio=3):
        super(ECB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.conv = LocalityFeedForward(
            out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

        self.norm = norm_layer(out_channels)
        # self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio,
        # drop=drop, bias=True)
        # self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        print("ECB:")
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        print("after sum: ", x.shape)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        # x = x + self.mlp_path_dropout(self.mlp(out))
        x = x + self.conv(out)
        print("after sum: ", x.shape)
        print("----------")
        return x


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(self, dim,
                 out_dim=None,
                 head_dim=32,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(
                kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        print("E_MHSA:")
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(
            C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 1, 3)
        print("q: ", q.shape)
        print("k: ", k.shape)
        print("v: ", v.shape)
        attn = (q @ k) * self.scale
        print("after attn: ", attn.shape)

        attn = attn.softmax(dim=-1)
        print("after softmax: ", attn.shape)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print("after reshape: ", attn.shape)
        x = self.proj(x)
        print("after projection", x.shape)
        x = self.proj_drop(x)
        print("----------")
        return x


class LTB(nn.Module):
    """
    Local Transformer Block
    """

    def __init__(self, in_channels,
                 out_channels,
                 path_dropout,
                 stride=1,
                 sr_ratio=1,
                 mlp_ratio=2,
                 head_dim=32,
                 mix_block_ratio=0.75,
                 attn_drop=0,
                 drop=0):
        super(LTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm3d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32)
        # print("out_channels: ", out_channels, "mhsa_out_channels: ", self.mhsa_out_channels)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels
        # print("mhsa_out_channels: ", self.mhsa_out_channels)
        # print("------------------")

        self.patch_embed = PatchEmbed(
            in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim,
                             sr_ratio=sr_ratio,
                             attn_drop=attn_drop,
                             proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(
            out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

        # self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        # self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        # PatchEmbed
        print("LTB:")
        # print("input_size: ", x.shape)
        x = self.patch_embed(x)
        B, C, H, W, D = x.shape
        print("B: ", B, "C: ", C, "H: ", H, "W: ", W, "D: ", D)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            # norm1
            out = self.norm1(x)
        else:
            out = x
        print("after norm1: ", out.shape)
        out = rearrange(out, "b c h w d -> b (h w d) c")  # b n c
        print("after rearrange: ", out.shape)
        # E_MHSA -> DropPath
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        print("after E_MHSA and DropPath: ", out.shape)
        x = x + rearrange(out, "b (h w d) c -> b c h w d", h=H, w=W, d=D)
        print("after rearrange: ", x.shape)

        out = self.projection(x)
        print("after projection: ", out.shape)
        out = out + self.mhca_path_dropout(self.mhca(out))
        print("after E_MHSA and DropPath: ", out.shape)
        x = torch.cat([x, out], dim=1)
        print("after concatenation: ", x.shape)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        print("after norm2: ", out.shape)
        x = x + self.conv(out)
        print("after LFFN: ", out.shape)
        # x = x + self.mlp_path_dropout(self.mlp(out))
        print("----------")
        return x


class MedViT(nn.Module):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0,
                 num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32,
                 mix_block_ratio=0.75,
                 use_checkpoint=False):
        super(MedViT, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [192],
                                   [384, 384, 384, 384, 384] *
                                   (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [768]]

        # Next Hybrid Strategy
        self.stage_block_types = [[ECB] * depths[0],
                                  [ECB] * (depths[1] - 1) + [LTB],
                                  [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
                                  [ECB] * (depths[3] - 1) + [LTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(1, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECB:
                    layer = ECB(input_channel, output_channel, stride=stride,
                                path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is LTB:
                    layer = LTB(input_channel, output_channel,
                                path_dropout=dpr[idx + block_id],
                                stride=stride,
                                sr_ratio=sr_ratios[stage_id],
                                head_dim=head_dim,
                                mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm3d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [
            sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        print('initialize_weights...')
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, ECB) or isinstance(module, LTB):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm,
                              nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print("MedViT:")
        print("input size: ", x.shape)
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        print("----------")
        return x


@register_model
def MedViT_small_3D(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedViT(stem_chs=[64, 32, 64], depths=[
                   3, 4, 10, 3], path_dropout=0.1, **kwargs)
    return model


@register_model
def MedViT_base_3D(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedViT(stem_chs=[64, 32, 64], depths=[
                   3, 4, 20, 3], path_dropout=0.2, **kwargs)
    return model


@register_model
def MedViT_large_3D(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedViT(stem_chs=[64, 32, 64], depths=[
                   3, 4, 30, 3], path_dropout=0.2, **kwargs)
    return model