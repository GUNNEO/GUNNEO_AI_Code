import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from image_encoder_backbone import vit as main_model, multi_fusion as fusion_model, pretrained_vit


def return_pretrained_model(model_name):
    pretrained_model = {
        # pretrained text model

        # pretrained vision model
        "ViT-B/16": [pretrained_vit.vit_b_16, 196, 768],
        "ViT-B/32": [pretrained_vit.vit_b_32, 49, 768],
        "ViT-L/16": [pretrained_vit.vit_l_16, 196, 1024],
        "ViT-L/32": [pretrained_vit.vit_l_32, 49, 1024]
    }
    return pretrained_model[model_name]


class CLIP(nn.Module):
    def __init__(
        self,
        num_img_modalities: int,
        embed_dim: int,
        # vision
        vision_pretrained: dict,
        image_size: int,
        patch_size: int,
        vision_channels: int,
        vision_layers: int,
        vision_hidden_dim: int,
        # text
        text_pretrained: dict,
        context_length: int,
        vocab_size: int,
        text_layers: int,
        text_headers: int,
        text_hidden_dim: int,

        # dropout and optional params
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        output_patches: Optional[int] = None
    ):
        super().__init__()
        self.context_length = context_length
        self.vision_pretrained = vision_pretrained
        self.text_pretrained = text_pretrained

        # set up encoder for images
        torch._assert(vision_hidden_dim % 64 == 0,
                      f"vision hidden dimension {vision_hidden_dim} indivisble by 64")
        vision_heads = vision_hidden_dim // 64
        self.output_patches = output_patches
        self.vision_hidden_dim = vision_hidden_dim
        if vision_pretrained["pretrained"]:
            model_info = return_pretrained_model(
                vision_pretrained["model_name"])
            # Load the pre-trained model
            pretrained_model = model_info[0](weights=None)
            state_dict = torch.load(
                vision_pretrained["model_path"], weights_only=True)
            pretrained_model.load_state_dict(state_dict, strict=False)
            self.vision_encoder = nn.ModuleList(
                [pretrained_model for _ in range(num_img_modalities)])
            self.output_patches = model_info[1]
            self.vision_hidden_dim = model_info[2]
        else:
            self.vision_encoder = nn.ModuleList([
                main_model.ViT(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_channels=vision_channels,
                    num_layers=vision_layers,
                    num_heads=vision_heads,
                    hidden_dim=vision_hidden_dim,
                    dropout=dropout,
                    attention_dropout=attn_dropout,
                    output_patches=output_patches if output_patches is not None else None
                ) for _ in range(num_img_modalities)
            ])

        # set up fusion model
        self.vision_fusion = fusion_model.FeatureFusion(
            num_patches=self.output_patches + 1 if self.output_patches is not None else (
                (image_size // patch_size) ** 2 + 1),
            num_dim=self.vision_hidden_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        # set up text trnsformer
        self.vocab_size = vocab_size
        # convert single token into a fixed length vector(length: text_hidden_dim)
        self.token_embedding = nn.Embedding(self.vocab_size, text_hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(self.context_length, text_hidden_dim))
        self.ln_final = main_model.LayerNorm(text_hidden_dim)
        self.text_projection = nn.Parameter(
            torch.empty(text_hidden_dim, embed_dim))
        # scale params
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_transformer = main_model.Encoder(
            num_layers=text_layers,
            num_heads=text_headers,
            hidden_dim=text_hidden_dim,
            mlp_dim=text_hidden_dim * 4,
            dropout=dropout,
            attention_dropout=attn_dropout,
            attn_mask=self.attn_mask()
        )

        # init weights for better optimazation
        self.initialize_parameters()

    def attn_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

        # text_transformer weights init
        proj_std = (self.text_transformer.hidden_dim ** -0.5) * \
            ((2 * self.text_transformer.num_layers) ** -0.5)
        attn_std = self.text_transformer.hidden_dim ** -0.5
        fc_std = (2 * self.text_transformer.hidden_dim) ** -0.5
        for layer in self.text_transformer.layers:
            nn.init.normal_(layer.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(layer.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(layer.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(layer.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.text_transformer.hidden_dim ** -0.5)

    @property
    def dtype(self):
        proj = self.vision_encoder[0].conv_proj
        if isinstance(proj, nn.Sequential):
            return proj[-1].weight.dtype
        else:
            return proj.weight.dtype

    def encode_image(self, image: torch.Tensor, modality_index: int):
        return self.vision_encoder[modality_index](image.type(self.dtype))

    def encode_text(self, text: torch.Tensor):
        torch._assert(len(text.shape) == 2,
                      "the input text shape not in this form (b, n_ctx)")
        # (b, n_ctx, hidden_dim)
        x = self.token_embedding(text).type(
            self.dtype)

        x = x + self.position_embedding.type(self.dtype)
        x = self.text_transformer(x)
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, text_transformer.hidden_dim]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
              ] @ self.text_projection

        return x

    def forward(
            self,
            images: torch.Tensor,
            text: torch.Tensor
    ):
        # image: (m, b, c, h, w)
        torch._assert(len(images.shape) == 5,
                      "the images input not in this form (m, b, c, h, w)")
        multi_image_features = []
        for i in range(len(images)):
            multi_image_features.append(self.encode_image(images[i], i))
        multi_image_features = torch.stack(
            multi_image_features)  # (m, b, n_p, d)

        # image features fusion
        image_features = self.vision_fusion(multi_image_features)  # (b d)

        # encode text
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(layer):

        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "final_weight"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
