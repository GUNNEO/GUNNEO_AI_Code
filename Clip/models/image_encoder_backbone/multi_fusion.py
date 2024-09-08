import torch
import torch.nn as nn


class BatchNorm(nn.BatchNorm1d):
    '''Temporarily convert precision to fp32'''

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # inherit the forward method from torch.nn.BatchNorm3d
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# element-wise summation, element-wise product, element-wise maximum
class FeatureFusion(nn.Module):
    def __init__(
        self,
        output_patches: int,
        output_dim: int,
        dropout: float = 0.0

    ):
        super().__init__()
        self.output_patches = output_patches
        self.output_dim = output_dim

        self.linear_1 = nn.Linear(self.output_patches * 3, self.output_patches)
        self.linear_2 = nn.Linear(self.output_patches, self.output_patches * 3)
        self.gelu = QuickGELU()
        self.bn_1 = BatchNorm(self.output_patches)
        self.bn_2 = BatchNorm(self.output_patches * 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # element-wise summation
        summation_x = torch.sum(x, dim=0)
        summation_x = torch.squeeze(summation_x, dim=0)
        summation_x = self.bn_1(summation_x)

        # element-wise product
        product_x = x[0]
        for index, element in enumerate(x):
            if index != 0:
                product_x = torch.mul(product_x, element)
        product_x = self.bn_1(product_x)

        # element-wise maximum
        maximum_x = x[0]
        for index, element in enumerate(x):
            if index != 0:
                maximum_x = torch.maximum(maximum_x, element)
        maximum_x = self.bn_1(maximum_x)

        # concat the output
        concat_x = torch.cat((summation_x, product_x, maximum_x), dim=1)

        # fusion weights
        x = concat_x.permute(0, 2, 1)
        x = self.linear_1(x).permute(0, 2, 1)
        x = self.bn_1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.linear_2(x).permute(0, 2, 1)
        x = self.bn_2(x)
        x = self.dropout(x)
        x = self.gelu(x)

        # concat fusion weights and ouput info
        x = torch.mul(concat_x, x)
        x = concat_x + x
        x = self.linear_1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.bn_1(x)
        x = self.dropout(x)

        return x


# (num_modalities, n, output_patches, output_dim)
x = torch.randn(4, 10, 246, 1024)
model = FeatureFusion(output_patches=246, output_dim=768)
out = model(x)
print(out.shape)
