import torch

x = torch.arange(64)[None, :]
a = torch.rand(127, 64)
b = torch.rand(64, 50)
c = a[b.long()]
mask_slice = slice(0, 1)
mask = torch.randn(32, 4, 256, 256)
mask = mask[:, mask_slice, :, :]
print(mask.shape)
