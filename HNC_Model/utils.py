import os
import nibabel as nib
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import torch
from einops import rearrange


class raiseException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        mlp_dim = 2048
        for _ in range(depth):
            # print (dim, mlp_dim)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Model(nn.Module):
    def __init__(self, img_h, img_d, patch_h, patch_d, num_classes, dim, depth, heads, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = (img_h // patch_h) * \
            (img_h // patch_h) * (img_d // patch_d)
        patch_dim = channels * patch_h ** 2 * patch_d

        self.patch_h = patch_h
        self.patch_d = patch_d

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # print (mlp_dim)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        # print (dim)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.Dropout(dropout)
        )

    def forward(self, img):
        ph = self.patch_h
        pd = self.patch_d
        # print(img.shape)
        x = rearrange(
            img, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=ph, p2=ph, p3=pd)
        # print(x.shape)
        x = self.patch_to_embedding(x)
        # print(x.shape)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)
        # print(self.pos_embedding.shape)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def get_image_transforms(type):
    with open("/Users/gunneo/Codes/AI/HNC_Model/means_stds.json") as f:
        means_stds = json.load(f)
        return transforms.Compose(
            [
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[means_stds[type + "_mean"]],
                    std=[means_stds[type + "_std"]],
                ),
            ]
        )


def preprocessing_mri(mri_path, mask_path, type):
    mri_image = nib.load(mri_path)
    mri = mri_image.get_fdata()
    mask_image = nib.load(mask_path)
    mask = mask_image.get_fdata()

    h1, w1, d1 = mri.shape
    h2, w2, d2 = mask.shape

    if h1 != h2 or w1 != w2 or d1 != d2:
        print(mri.shape)
        print(mask.shape)
        print(mri_path)
        raise raiseException(
            "Inconsistent dimensions between mask and original image")

    preprocess = get_image_transforms(type)

    d0 = 30
    image_tensor = torch.zeros((1, 320, 320, d0))

    if d1 >= d0:
        slice_indices = np.linspace(
            0, d0 - 1, d0, dtype=int)
        for i, idx in enumerate(slice_indices):
            slice_mri = mri[:, :, idx]
            slice_mask = mask[:, :, idx]
            slice_roi = np.multiply(slice_mri, slice_mask)
            slice_roi = Image.fromarray(slice_roi).convert("L")
            slice_tensor = preprocess(slice_roi)
            image_tensor[:, :, :, i] = slice_tensor
    else:
        slice_indices = np.linspace(0, d1 - 1, d1, dtype=int)
        for i, idx in enumerate(slice_indices):
            slice_mri = mri[:, :, idx]
            slice_mask = mask[:, :, idx]
            slice_roi = np.multiply(slice_mri, slice_mask)
            slice_roi = Image.fromarray(slice_roi).convert("L")
            slice_tensor = preprocess(slice_roi)
            image_tensor[:, :, :, i] = slice_tensor
        for j in range(d0 - d1):
            image_tensor[:, :, :, d1 + j - 1] = image_tensor[:, :, :, d1 - 1]
    return image_tensor


def check_all_mri(mri_path, mask_path):
    for name in os.listdir(mri_path):
        single_mri_path = os.path.join(mri_path, name)
        single_mask_path = os.path.join(mask_path, name)
        for file in os.listdir(single_mri_path):
            if file == "T1+C COR.nii.gz":
                path1 = os.path.join(single_mri_path, file)
                path2 = os.path.join(single_mask_path, file)
                img = preprocessing_mri(path1, path2, "T1+C COR")
                img = img[np.newaxis, :]
                print(img.shape)
                return


mri_path = '/Users/gunneo/Documents/Dataset/370/image-before/'
mask_path = '/Users/gunneo/Documents/Dataset/370/ROI/'
# T2: checked T1+C: checked T1+C COR: checked
# T1+C COR:
# min: 320, 320, 15
# max: 640, 640, 110
# T1+C:
# min: 256, 208, 20
# max: 704, 674, 160
# T2:
# min: 320, 270, 25
# max: 864, 864, 90
