import torch
import torch.nn as nn


class BatchNorm1d(nn.BatchNorm1d):
    '''Temporarily convert precision to fp32'''

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # inherit the forward method from torch.nn.BatchNorm1d
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class BatchNorm2d(nn.BatchNorm2d):
    '''Temporarily convert precision to fp32'''

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # inherit the forward method from torch.nn.BatchNorm2d
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class BatchNorm3d(nn.BatchNorm3d):
    '''Temporarily convert precision to fp32'''

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # inherit the forward method from torch.nn.BatchNorm3d
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


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
