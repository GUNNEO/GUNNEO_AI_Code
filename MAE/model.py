import torch
import torch.nn as nn
from typing import Optional
from timm.models.vision_transformer import Block
from timm.layers.helpers import to_2tuple
from pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        # return (img_size, img_size)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(
            W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        # N is the number of patches
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MAE(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        num_encoder_layers: int = 24,
        num_encoder_heads: int = 16,
        num_decoder_layers: int = 8,
        num_decoder_heads: int = 16,
        mlp_ratio: float = 4.,
        norm_pix_loss: bool = False,
        norm_layer: Optional[nn.Module] = nn.LayerNorm
    ):
        super().__init__()

        """
        MAE Encoder
        """
        # similar process of vit patch embedding
        self.PatchEmbed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.PatchEmbed.num_patches

        # define cls token, 1 is used for broadcasting
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # notice that the positional encoding do not attend the process of training
        self.encoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim), requires_grad=False)
        self.encoder = nn.ModuleList([Block(dim=embed_dim, num_heads=num_encoder_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, init_values=None, norm_layer=norm_layer) for _ in range(num_encoder_layers)])
        self.encoder_norm = norm_layer(embed_dim)

        """
        MAE Decoder
        """
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            (1, num_patches + 1, decoder_embed_dim)), requires_grad=False)
        self.decoder = nn.ModuleList([Block(dim=decoder_embed_dim, num_heads=num_decoder_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, init_values=None, norm_layer=norm_layer) for _ in range(num_decoder_layers)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels)

        self.norm_pix_loss = norm_pix_loss

        # init the weights
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1], int(self.PatchEmbed.num_patches**.5), cls_token=True)
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1], int(self.PatchEmbed.num_patches**.5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(
        #     torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.PatchEmbed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # rand return a tensor filled with random numbers from a uniform distribution on interval[0,1)
        noise = torch.rand(N, L, device=x.device)
        # argsort will sort the values from small to large and return the relative index of original tensor
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(
        self,
        imgs: torch.Tensor
    ):
        """
        convert the original images into the shape of (B x N x (patch_size x patch_size x3))
        """
        b, c, h, w = imgs.shape
        p = self.PatchEmbed.patch_size[0]
        torch._assert((h == w and h // p == 0),
                      "imgs' height and width are not equal or the shape can not be devided by the patch size")
        h, w = h // p, w // p
        x = torch.reshape(imgs, (b, c, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = torch.reshape(x, (b, h * w, p * p * c))
        return x

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ):
        # embedding the patch and add pos embedding
        x = self.PatchEmbed(x)
        x = x + self.encoder_pos_embed[:, 1:, :]

        # add random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor
    ):
        x = self.decoder_embed(x)

        # append mask tokens to sequence, cls token will not be processed in this phrase
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(
        self,
        ori_x: torch.Tensor,
        pred_x: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        ori_x: B x C x H x W
        pred_x: B x N x (patch_size x patch_size x 3)
        """
        target = self.patchify(imgs=ori_x)
        # determine whetehr we need to norm the target imgs
        if self.norm_pix_loss:
            mean = torch.mean(target, dim=-1, keepdim=True)
            var = torch.var(target, dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred_x - ori_x) ** 2
        loss_mean = torch.mean(loss, dim=-1)
        loss = (loss_mean * mask).sum()
        return loss

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ):
        latent_x, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred_x = self.forward_decoder(latent_x, ids_restore)
        loss = self.forward_loss(x, pred_x, mask)
        return loss, pred_x, mask


if __name__ == "__main__":
    model = MAE()
    # x = torch.randn(2, 3, 224, 224)
    # out = model(x)
