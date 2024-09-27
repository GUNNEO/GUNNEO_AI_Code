import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np
from transformers import BertConfig
from text_encoder_backbone.modeling_bert import BertModel
from image_encoder_backbone import vit as main_model, pretrained_vit, multi_fusion, attn_gan_fusion


def return_pretrained_model(model_name):
    pretrained_model = {
        # pretrained text model
        "ClinicalBERT": [768],
        "BlueBERT-B": [768],
        "BlueBERT-L": [1024],

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
        num_text_modalities: int,
        num_img_modalities: int,
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
        output_patches: Optional[int] = None,
        img_fusion_method: str = "attn_gan_fusion",
        conv_stem_configs: Optional[List[main_model.ConvStemConfig]] = None
    ):
        super().__init__()
        self.num_text_modalities = num_text_modalities
        self.num_img_modalities = num_img_modalities
        self.vision_channels = vision_channels
        self.context_length = context_length
        self.vision_pretrained = vision_pretrained
        self.text_pretrained = text_pretrained
        self.img_fusion_method = img_fusion_method

        """
        set up for image part
        """
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
            pretrained_model = model_info[0](
                num_input_channels=vision_channels, weights=None)
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
                    output_patches=output_patches if output_patches is not None else None,
                    conv_stem_configs=conv_stem_configs
                ) for _ in range(num_img_modalities)
            ])

        # set up fusion model for image encoder
        img_fusion_method_name = "multi_fusion, attn_gan_fusion"
        if img_fusion_method == "multi_fusion":
            self.vision_fusion = multi_fusion.FeatureFusion(
                num_patches=self.output_patches + 1 if self.output_patches is not None else (
                    (image_size // patch_size) ** 2 + 1),
                num_dim=self.vision_hidden_dim,
                output_dim=self.vision_hidden_dim // 2,
                dropout=dropout
            )
        elif img_fusion_method == "attn_gan_fusion":
            self.vision_fusion = attn_gan_fusion.FeatureFusion(
                num_modalities=self.num_img_modalities,
                input_dim=self.vision_hidden_dim
            )
        else:
            raise RuntimeError(
                f"invalid fusion method, expected {img_fusion_method_name} but got {img_fusion_method}")

        """
        set up for text part
        """
        if self.text_pretrained["pretrained"]:
            config_file = self.text_pretrained["config_path"]
            state_dict = torch.load(
                self.text_pretrained["model_path"], weights_only=True)
            config = BertConfig.from_json_file(config_file)
            self.text_transformer = nn.ModuleList([
                BertModel(config) for _ in range(num_text_modalities)])
            for model_index in range(num_text_modalities):
                self.text_transformer[model_index].load_state_dict(
                    state_dict, strict=False)
            text_hidden_dim = return_pretrained_model(
                self.text_pretrained["model_name"])[0]
            self.ln_final = main_model.LayerNorm(text_hidden_dim)
            self.text_projection = nn.ParameterList([nn.Parameter(
                torch.empty(text_hidden_dim, vision_hidden_dim)) for _ in range(num_text_modalities)])
        else:
            # set up text trnsformer
            self.vocab_size = vocab_size
            # convert single token into a fixed length vector(length: text_hidden_dim)
            self.token_embedding = nn.ModuleList([nn.Embedding(
                self.vocab_size, text_hidden_dim) for _ in range(num_text_modalities)])
            self.position_embedding = nn.ParameterList([nn.Parameter(
                torch.empty(self.context_length, text_hidden_dim)) for _ in range(num_text_modalities)])
            self.ln_final = main_model.LayerNorm(text_hidden_dim)
            self.text_projection = nn.ParameterList([nn.Parameter(
                torch.empty(text_hidden_dim, vision_hidden_dim)) for _ in range(num_text_modalities)])

            self.text_transformer = nn.ModuleList([main_model.Encoder(
                num_layers=text_layers,
                num_heads=text_headers,
                hidden_dim=text_hidden_dim,
                mlp_dim=text_hidden_dim * 4,
                dropout=dropout,
                attention_dropout=attn_dropout,
                attn_mask=self.attn_mask()
            ) for _ in range(num_text_modalities)])

        # set up fusion model for text encoder
        if num_text_modalities != 1:
            # using attn_gan_fusion as default
            self.text_fusion = attn_gan_fusion.FeatureFusion(
                num_modalities=num_text_modalities,
                input_dim=vision_hidden_dim
            )
        else:
            self.text_fusion = nn.Linear(
                vision_hidden_dim, vision_hidden_dim // 2)

            # scale params
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init weights for better optimazation
        if not self.text_pretrained["pretrained"]:
            self.initialize_parameters()

    def attn_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        for modality in range(self.num_text_modalities):
            nn.init.normal_(self.token_embedding[modality].weight, std=0.02)
            nn.init.normal_(self.position_embedding[modality], std=0.01)

            # text_transformer weights init
            cur_text_model = self.text_transformer[modality]
            proj_std = (cur_text_model.hidden_dim ** -0.5) * \
                ((2 * cur_text_model.num_layers) ** -0.5)
            attn_std = cur_text_model.hidden_dim ** -0.5
            fc_std = (2 * cur_text_model.hidden_dim) ** -0.5
            for layer in cur_text_model.layers:
                nn.init.normal_(layer.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(layer.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(layer.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(layer.mlp.c_proj.weight, std=proj_std)

            if self.text_projection is not None:
                nn.init.normal_(self.text_projection[modality],
                                std=cur_text_model.hidden_dim ** -0.5)

    @property
    def dtype(self):
        proj = self.vision_encoder[0].conv_proj if (self.vision_pretrained["pretrained"] and self.vision_channels == 3) or (
            not self.vision_pretrained["pretrained"]) else self.vision_encoder[0].modify_conv_proj
        if isinstance(proj, nn.Sequential):
            return proj[-1].weight.dtype
        else:
            return proj.weight.dtype

    def encode_image(
        self,
        image: torch.Tensor,
        modality_index: int
    ):
        return self.vision_encoder[modality_index](image.type(self.dtype))

    def encode_text(
        self,
        text: torch.Tensor,
        modality_index: int,
        mask: Optional[torch.Tensor] = None
    ):
        torch._assert(len(text.shape) == 2,
                      "the input text shape not in this form (b, n_ctx)")
        if self.text_pretrained["pretrained"]:
            x = self.ln_final(self.text_transformer[modality_index](
                text, attention_mask=mask, changed_type=self.dtype).pooler_output) @ self.text_projection[modality_index]
        else:
            # (b, n_ctx, hidden_dim)
            x = self.token_embedding[modality_index](text).type(
                self.dtype)

            x = x + self.position_embedding[modality_index].type(self.dtype)
            x = self.text_transformer[modality_index](x)
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, text_transformer.hidden_dim]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
                  ] @ self.text_projection[modality_index]

        return x

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ):
        # image: (b, n_m, c, h, w)
        torch._assert(len(images.shape) == 5,
                      "the images input not in this form (b, n_m, c, h, w)")
        torch._assert(images.shape[1] == self.num_img_modalities,
                      f"the number of input modalities is not equal to the defined number of modalities {self.num_img_modalities}, but got {images.shape[1]}")
        multi_image_features = []
        for m in range(self.num_img_modalities):
            image = images[:, m, :, :, :]
            multi_image_features.append(self.encode_image(image, m))
        multi_image_features = torch.stack(
            multi_image_features)  # (m, b, n_p + 1, d)

        # image features fusion
        d0_img, d1_img, ds_img = None, None, None
        if self.img_fusion_method == "multi_fusion":
            image_features = self.vision_fusion(multi_image_features)  # (b d)
        else:
            # extract cls token part
            multi_image_features = multi_image_features[:, :, 0, :]
            image_features, d0_img, d1_img, ds_img = self.vision_fusion(
                multi_image_features)

        # encode text
        d0_text, d1_text, ds_text = None, None, None
        multi_text_features = []
        for m in range(self.num_text_modalities):
            if masks is not None:
                mask = masks[m]
            else:
                mask = None
            text = texts[:, m, :]
            multi_text_features.append(self.encode_text(
                text=text, modality_index=m, mask=mask))
        multi_text_features = torch.stack(multi_text_features, dim=0)

        # text features fusion
        if self.num_text_modalities:
            text_features, d0_text, d1_text, ds_text = self.text_fusion(
                multi_text_features)
        else:
            text_features = self.fusion(text_features.squeeze(0))

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, d0_img, d1_img, ds_img, d0_text, d1_text, ds_text


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
