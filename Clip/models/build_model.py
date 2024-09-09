import torch
import models
from typing import Optional


def load_clip(
    num_img_modalities: int,
    embed_dim: int = 256,
    image_size: int = 224,
    patch_size: int = 16,
    vision_channels: int = 1,
    vision_layers: int = 12,
    vision_hidden_dim: int = 768,
    context_length: int = 77,
    vocab_size: int = 49408,
    text_layers: int = 12,
    text_headers: int = 8,
    text_hidden_dim: int = 768,
    output_patches: Optional[int] = 49
):
    params = locals()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CLIP(**params)
    model.to(device)
    models.convert_weights(model)
    return model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = torch.randn(4, 16, 1, 224, 224).to(device)
text = torch.randint(0, 49408, (16, 77)).to(device)
model = load_clip(num_img_modalities=4)
i, t = model(image, text)
print(i.shape, t.shape)
