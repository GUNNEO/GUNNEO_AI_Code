import torch
import torch.nn as nn
from utils import LayerNorm, BatchNorm1d, QuickGELU


# element-wise summation, element-wise product, element-wise maximum
class FeatureFusion(nn.Module):
    def __init__(
        self,
        num_patches: int,
        num_dim: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_dim = num_dim
        self.output_dim = output_dim

        self.linear_1 = nn.Linear(self.num_patches * 3, self.num_patches)
        self.linear_2 = nn.Linear(self.num_patches, self.num_patches * 3)
        self.gelu = QuickGELU()
        self.bn_1 = BatchNorm1d(self.num_patches)
        self.bn_2 = BatchNorm1d(self.num_patches * 3)
        self.ln = LayerNorm(self.num_dim)
        self.dropout = nn.Dropout(dropout)
        scale = num_dim ** -0.5
        self.final_weight = nn.Parameter(
            scale * torch.randn(self.num_dim, self.output_dim))

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

        # layernorm for the final output with additional final projection weights
        x = self.ln(x[:, 0, :])
        x = x @ self.final_weight

        return x
