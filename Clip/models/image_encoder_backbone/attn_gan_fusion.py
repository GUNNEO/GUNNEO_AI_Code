import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import QuickGELU, BatchNorm1d


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class FeatureFusion(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        input_dim: int
    ):
        super().__init__()
        self.num_modalities = num_modalities
        # create
        self.multi_linear = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_modalities)])

        cut_dim = input_dim // 2
        self.params_w = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, cut_dim)) for _ in range(num_modalities)])
        self.params_attn_w = nn.ParameterList(
            [nn.Parameter(torch.randn(cut_dim, cut_dim)) for _ in range(num_modalities)])
        self.param_w = nn.Parameter(torch.randn(input_dim, cut_dim))
        self.params_b = nn.ParameterList(
            [nn.Parameter(torch.randn(cut_dim)) for _ in range(num_modalities)])
        self.params_attn_b = nn.ParameterList(
            [nn.Parameter(torch.randn(cut_dim)) for _ in range(num_modalities)])
        self.param_b = nn.Parameter(torch.randn(cut_dim))

        self.gelu = QuickGELU()
        self.tanh = nn.Tanh()
        self.bn = BatchNorm1d(cut_dim)

        self.D0 = nn.ModuleList([self._build_discriminator(
            cut_dim, num_modalities) for _ in range(num_modalities)])
        self.D1 = nn.ModuleList([self._build_discriminator(
            cut_dim, num_modalities) for _ in range(num_modalities)])
        self.Ds = nn.ModuleList([self._build_discriminator(
            cut_dim, num_modalities) for _ in range(num_modalities)])
        for m in range(num_modalities):
            self.Ds[m].load_state_dict(self.D1[m].state_dict())

    def _build_discriminator(self, input_dim, num_modalities):
        layers = []
        cur_dim = input_dim
        while True:
            next_dim = cur_dim // 2
            if next_dim > 128:
                layers.append(nn.Linear(cur_dim, next_dim))
                layers.append(QuickGELU())
                cur_dim = next_dim
            else:
                layers.append(nn.Linear(cur_dim, 128))
                layers.append(QuickGELU())
                layers.append(nn.Linear(128, num_modalities))
                break
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor
    ):
        # input shape (n_m, b, d)
        torch._assert(len(x.shape) == 3,
                      "input shape should be in this form (n_m, b, d)")
        torch._assert(x.shape[0] == self.num_modalities,
                      f"input tensor should have {self.num_modalities} but got {x.shape[0]} modalities")

        rep = []
        d0 = []
        d1 = []
        ds = []
        for m0 in range(self.num_modalities):
            s = []
            c = []
            d0_m0 = []
            d1_m0 = []
            ds_m0 = []
            for m1, modality_tensor in enumerate(x):
                e_m1 = self.multi_linear[m0](modality_tensor)
                e_m1_detached = e_m1.detach()

                s_m1 = self.gelu(
                    self.bn(e_m1_detached @ self.params_w[m0] + self.params_b[m0]))

                attn_m1 = self.tanh(
                    self.bn(s_m1 @ self.params_attn_w[m0] + self.params_attn_b[m0]))

                s.append(attn_m1 * s_m1)

                c_m1 = self.gelu(self.bn(e_m1 @ self.param_w + self.param_b))
                c_m1_detached = c_m1.detach()

                d0_logits = self.D0[m0](c_m1_detached)
                d0_prob_m1 = F.softmax(d0_logits, dim=1)
                d0_m0.append(d0_prob_m1)

                w_m1 = (1 - d0_prob_m1[:, m1]).unsqueeze(1) * s_m1
                c.append(w_m1)

                w_m1_grl = grad_reverse(w_m1, alpha=1.0)
                d1_logits = self.D1[m0](w_m1_grl)
                d1_prob_m1 = F.softmax(d1_logits, dim=1)
                d1_m0.append(d1_prob_m1)

                ds_logits = self.Ds[m0](s_m1)
                ds_prob_m1 = F.softmax(ds_logits, dim=1)
                ds_m0.append(ds_prob_m1)
            d0.append(torch.stack(d0_m0, dim=0))
            d1.append(torch.stack(d1_m0, dim=0))
            ds.append(torch.stack(ds_m0, dim=0))
            c, _ = torch.max(torch.stack(c, dim=0), dim=0)
            rep.append(sum(s) + c)

        d0 = torch.stack(d0, dim=0)
        d1 = torch.stack(d1, dim=0)
        ds = torch.stack(ds, dim=0)

        return sum(rep), d0, d1, ds


def compute_losses(
    d0: torch.Tensor,
    d1: torch.Tensor,
    ds: torch.Tensor,
):
    """
    compute D0, D1, Ds loss

    args：
    - d0: [M, M, B, M]
    - d1: [M, M, B, M]
    - ds: [M, M, B, M]
    - num_modalities: M
    """
    M = d0.shape[-1]
    B = d0.shape[2]

    device = d0.device

    modality_labels = torch.arange(M).unsqueeze(1).unsqueeze(
        2).repeat(1, M, B).to(device)  # [M, M, B]

    # shape [M * M * B, M]
    d0_logits = d0.view(-1, M)
    d1_logits = d1.view(-1, M)
    ds_logits = ds.view(-1, M)

    labels = modality_labels.view(-1)  # [M * M * B]

    D0_loss = F.cross_entropy(d0_logits, labels)
    D1_loss = F.cross_entropy(d1_logits, labels)
    Ds_loss = F.cross_entropy(ds_logits, labels)

    return D0_loss, D1_loss, Ds_loss
