"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pos):
        '''
        pos - b x n x d
        '''
        b, n, d = pos.shape
        y_embed = pos[:, :, 1]  # b x n
        x_embed = pos[:, :, 0]
        if self.normalize:
            eps = 1e-6
            y_embed = torch.clamp(y_embed / (y_embed.max() + eps), 0, 1) * self.scale
            x_embed = torch.clamp(x_embed / (x_embed.max() + eps), 0 , 1) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)  # npf
        dim_t = self.temperature ** (2 * (dim_t.div(2, rounding_mode='floor')) / self.num_pos_feats)  # npf

        pos_x = x_embed[:, :, None] / dim_t  # b x n x npf
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.cat(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2
        )
        pos_y = torch.cat(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=2
        )
        pos = torch.cat((pos_x, pos_y), dim=2)  # b x n x d'
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def get_2dpos_of_curr_ps_in_min_ps(height, width, patch_size, min_patch_size, scale):
    patches_coords = torch.meshgrid(torch.arange(0, width // min_patch_size, patch_size // min_patch_size), torch.arange(0, height // min_patch_size, patch_size // min_patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.transpose(0, 1)
    patches_coords = patches_coords.reshape(-1, 2)
    n_patches = patches_coords.shape[0]

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = rearrange(x.transpose(1, 2), 'b c (h w) -> b c h w', b=B, c=C, h=H, w=W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, dw_conv=True, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        if dw_conv:
            self.dwconv = DWConv(hidden_dim)
        self.dw_conv = dw_conv
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, h, w):
        x = self.fc1(x)
        if self.dw_conv:
            x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, h, w):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, layer_scale=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, h, w):
        y = self.attn(self.norm1(x), h, w)
        if not self.layer_scale:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        else:
            x = x + self.drop_path(self.gamma1 * y)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), h, w))
        if torch.isnan(x).any():
            print("NaNs detected in ff-attn in ViT")
        return x

class DownSampleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.g_norm = nn.GroupNorm(1, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.g_norm(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Parameter):
        nn.init.trunc_normal_(m, std=0.02)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, channels):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, channels):
        super().__init__()

        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dims = [int(embed_dim // 2**(n_layers -1 - i)) for i in range(n_layers) ]
        emb_dim_list = [channels] + emb_dims
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], emb_dim_list[i + 1])
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, im):
        x = self.conv_layers(im).flatten(2).transpose(1, 2)
        x = self.out_norm(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
            self,
            n_blocks,
            dim,
            n_heads,
            dim_ff,
            dropout=0.0,
            drop_path_rate=[0.0],
            layer_scale=0.0
    ):
        super().__init__()

        # transformer blocks
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, dim_ff, dropout, drop_path_rate[i], layer_scale) for i in range(n_blocks)]
        )

    def forward(self, x, h, w):
        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x, h, w)
        return x


class MixResViT(nn.Module):
    def __init__(
            self,
            patch_sizes,
            n_layers,
            d_model,
            n_heads,
            mlp_ratio=4.0,
            dropout=0.0,
            drop_path_rate=[0.0],
            channels=3,
            split_ratio=4,
            n_scales=2,
            min_patch_size=4,
            upscale_ratio=0.0,
            first_layer=True,
            layer_scale=0.0,
            num_register_tokens=0,
            out_features=['res5']
    ):
        super().__init__()
        self.patch_size = patch_sizes[-1]
        self.patch_sizes = patch_sizes

        self.patch_size = self.patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.min_patch_size = min_patch_size
        self.upscale_ratio = upscale_ratio
        self.first_layer = first_layer
        self.num_register_tokens = num_register_tokens

        num_features = d_model
        self.num_features = num_features
        self._out_features = out_features

        if self.first_layer:
            # Pos Embs
            self.pe_layer = PositionEmbeddingSine(d_model // 2, normalize=True)
            self.patch_embed = OverlapPatchEmbedding(
                self.patch_size,
                d_model,
                channels,
            )
        else:
            self.token_norm = nn.LayerNorm(channels)
            if channels != d_model:
                self.token_projection = nn.Linear(channels, d_model)
            else:
                self.token_projection = nn.Identity()
        dim_ff = int(d_model * mlp_ratio)
        # transformer layers
        self.layers = TransformerLayer(n_layers, d_model, n_heads, dim_ff, dropout, drop_path_rate, layer_scale)

        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, d_model)) if num_register_tokens else None
        )

        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()
        self.norm_out = nn.LayerNorm(d_model)
        self.apply(init_weights)

        print("Successfully built MixResViT model!")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, scale, features, features_pos, upsampling_mask, mask_ratio):
        B, _, H, W = im.shape
        PS = self.patch_size
        patched_im_size = (H // PS, W // PS)
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)
        outs = {}
        if self.first_layer:
            x = self.patch_embed(im)
            if torch.isnan(x).any():
                print("NaNs detected in patch-embedded features in ViT in scale {}".format(scale))
            pos = get_2dpos_of_curr_ps_in_min_ps(H, W, PS, self.min_patch_size, scale).to('cuda')
            pos = pos.repeat(B, 1, 1)
            #print("Encoder pos max x: {}, max y: {}, and all pos: {}".format(pos[:, :, 0].max(), pos[:, :, 1].max(), pos))
            #self.test_pos_cover_and_overlap(pos[0], H, W, scale)
            pos_embed = self.pe_layer(pos[:,:,1:])
            x = x + pos_embed
            if torch.isnan(x).any():
                print("NaNs detected in pos-embedded features in ViT in scale {}".format(scale))
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            outs['mask'] = mask
            outs['ids_restore'] = ids_restore
        else:
            features = self.token_norm(features)
            x = self.token_projection(features)
            pos = features_pos
            if torch.isnan(x).any():
                print("NaNs detected in projected features in ViT in scale {}".format(scale))

        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(B, -1, -1), x], dim=1)
        x = self.layers(x, h=patched_im_size[0], w=patched_im_size[1])
        x = x[:, self.num_register_tokens:]
        orig_dtype = x.dtype
        out_name = self._out_features[0]
        outs[out_name] = self.norm_out(x.float()).to(orig_dtype)
        outs[out_name + "_pos"] = pos[:,:,1:]  # torch.div(pos_scale, 2 ** (self.n_scales - s - 1), rounding_mode='trunc')
        outs[out_name + "_spatial_shape"] = patched_im_size
        outs[out_name + "_scale"] = pos[:, :, 0]
        outs["min_spatial_shape"] = min_patched_im_size
        return outs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore