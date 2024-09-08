import torch
import torch.nn as nn
from image_encoder_backbone import vit, multi_fusion
from text_encoder_backbone import tokenizer


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
    ):
        super().__init__()
