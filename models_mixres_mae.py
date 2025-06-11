# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed
from modeling.mixres_vit import MixResViT
from modeling.mixres_neighbour import MixResNeighbour

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class UpDownBackbone(nn.Module):
    def __init__(self, backbones, backbone_dims, all_out_features, n_scales, bb_in_feats, img_size=224, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, norm_pix_loss=False):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        final_upsampling_ratios = []
        for b in self.backbones:
            final_upsampling_ratios.append(b.upscale_ratio)
        self.final_upsampling_ratios = final_upsampling_ratios
        self.backbone_dims = backbone_dims
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}
        self.n_scales = n_scales
        self.bb_in_feats = bb_in_feats
        scales = list(range(self.n_scales))
        self.bb_scales = scales + scales[-2::-1]

        # MAE decoder specifics
        embed_dim = backbone_dims[-1]
        patch_size = self.backbones[-1].patch_size
        num_patches = (img_size // patch_size) ** 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True) # decoder to patch
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.apply(self._init_weights)

        #print("Successfully built UpDownBackbone model!")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, im, mask_ratio=0.75):
        B, C, H, W = im.shape
        #print("Input image shape is {}".format(im.shape))
        up = True
        upsampling_mask = None
        features = None
        features_pos = None
        outs = {}
        for j in range(len(self.backbones)):
            scale = self.bb_scales[j]
            output = self.backbones[j](im, scale, features, features_pos, upsampling_mask, mask_ratio)
            if j == 0:
                mask = output['mask']
                ids_restore = output['ids_restore']
            all_feat = []
            all_scale = []
            all_pos = []
            all_ss = []
            bb_out_features = self.backbones[j]._out_features
            for i, f in enumerate(bb_out_features):
                feat = output[f]
                feat_pos = output[f + '_pos']
                feat_scale = output[f + '_scale']
                feat_ss = output[f + '_spatial_shape']
                #print("For layer {} with spatial shape {}, feat_pos shape: {}, max: {}".format(j, feat_ss, feat_pos.shape, feat_pos.max()))
                B, N, C = feat.shape
                if f + '_pos' in outs:
                    pos_indices = self.find_pos_org_order(outs[f + '_pos'], feat_pos)
                    b_ = torch.arange(B).unsqueeze(-1).expand(-1, N)
                    feat = feat[b_, pos_indices]
                    feat_pos = feat_pos[b_, pos_indices]
                    feat_scale = feat_scale[b_, pos_indices]
                    assert (outs[f + '_pos'] == feat_pos).all()
                    outs[f].append(feat)
                else:
                    outs[f] = [feat]
                    outs[f + '_pos'] = feat_pos
                    outs[f + '_scale'] = feat_scale
                    outs[f + '_spatial_shape'] = feat_ss
                if f in self.bb_in_feats[j + 1]:
                    if j >= self.n_scales - 1:
                        out_feat = torch.cat(outs[f][-((j - self.n_scales + 1)*2 + 2):], dim=2)
                    else:
                        out_feat = feat
                    #print("For bb level {}, feature {} shape is {}".format(j, f, out_feat.shape))
                    all_feat.append(out_feat)
                    all_pos.append(feat_pos)
                    all_scale.append(feat_scale)
                    all_ss.append(feat_ss)
            if j == self.n_scales - 1:
                up = False
            if up:
                B, N, C = all_feat[0].shape
                upsampling_mask = self.generate_random_upsampling_mask(B, N, feat.device)
                #upsampling_mask = self.upsamplers[scale](all_feat[0]).squeeze(-1)

            #print("Upsampling mask for scale {}: pred: {}, oracle: {}".format(scale, upsampling_mask_pred.shape, upsampling_mask_oracle.shape))

            if j < len(self.backbones) - 1:
                all_pos = torch.cat(all_pos, dim=1)
                all_scale = torch.cat(all_scale, dim=1)
                features_pos = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
                features = torch.cat(all_feat, dim=1)
                #print("For bb level {}, feature shape is {}".format(j, features.shape))

        x = output[self.all_out_features[-1]]
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        patch_size = self.backbones[-1].patch_size
        target = self.patchify(imgs, patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


    def generate_random_upsampling_mask(self, batch_size, n_tokens, device):
        upsampling_mask = torch.randn(batch_size, n_tokens).float().to(device)
        return upsampling_mask
    def find_pos_org_order(self, pos_org, pos_shuffled):
        dists = torch.cdist(pos_org.float(), pos_shuffled.float(), p=1)  # Manhattan distance
        pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

        return pos_indices

    def generate_max_norm_upsampling_mask(self, features):
        upsampling_mask = features.norm(dim=2)
        return upsampling_mask


    def get_mask_and_ids_restore(self, grid_pos, pos):
        B, N, _ = grid_pos.shape
        B, M, _ = pos.shape
        mask_len = N - M
        mask_imp = (grid_pos.unsqueeze(2) == pos.unsqueeze(1)).all(-1).any(-1)
        idx_dummy = torch.cumsum(~mask_imp, dim=1) - 1
        idx_imp = torch.cumsum(mask_imp, dim=1) - 1 + mask_len
        ids_restore = torch.where(mask_imp, idx_imp, idx_dummy).to(pos.device)

        mask = torch.zeros([B, N], device=pos.device)
        mask[:, :mask_len] = 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

        h = w = imgs.shape[2] // patch_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
        return x

    def unpatchify(self, x, patch_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
        return imgs


def get_2dpos_of_curr_ps_in_min_ps(height, width, patch_size, min_patch_size, batch_size):
    patches_coords = torch.meshgrid(torch.arange(0, width // min_patch_size, patch_size // min_patch_size), torch.arange(0, height // min_patch_size, patch_size // min_patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.transpose(0, 1)
    patches_coords = patches_coords.reshape(-1, 2)
    patches_coords = patches_coords.repeat(batch_size, 1, 1)

    return patches_coords

def mae_mixres_small_patch32_dec512d8b(**kwargs):
    bb_in_feats = [[None], ["res5"], ["res5", "res4"], ["res5", "res4", "res3"], ["res5", "res4", "res3"],
                   ["res5", "res4"], ["res5"], [None]]
    all_backbones = []
    bb_names = ['MixResViT', 'MixResNeighbour', 'MixResNeighbour', 'MixResNeighbour', 'MixResNeighbour', 'MixResNeighbour', 'MixResViT']
    n_scales = 4
    n_layers = len(bb_names)
    c_embed_dims = [512, 256, 128, 64, 128, 256, 512]
    c_depths = [1,1,1,1,1,1,1]
    c_num_heads = [ 16, 8, 4, 2, 4, 8, 16]
    c_patch_sizes = [32, 16, 8, 4, 8, 16, 32]
    c_drop_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    c_attn_drop_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    c_upscale_ratios = [0.0, 0.85, 0.65, 0.5, 0.0, 0.0, 0.0]
    c_split_ratios = [4, 4, 4, 4, 4, 4, 4]
    c_mlp_ratios = [3., 3., 3., 3., 3., 3., 3.]
    c_cluster_sizes = [8, 8, 8, 8, 8, 8, 8]
    c_nbhd_sizes = [48,48,48,48,48,48,48]
    c_out_features = ["res2", "res3", "res4", "res5"]
    c_keep_old_scale = True
    c_add_image_data_to_all = False
    c_drop_path_rate = 0.1
    c_layer_scale = 0.0
    c_register_tokens = 0
    min_patch_size = c_patch_sizes[n_layers // 2]
    for layer_index, name in enumerate(bb_names):
        if layer_index == 0:
            first_layer = True
            in_chans = 3
        else:
            first_layer = False
            in_chans = c_embed_dims[layer_index - 1]
        if layer_index >= n_scales:
            scale = n_layers - layer_index - 1
            patch_sizes = c_patch_sizes[layer_index:]
            out_features = c_out_features[-(n_layers - layer_index):]
            in_chans = sum(c_embed_dims[-(layer_index + 1):-(n_layers - layer_index)])
        else:
            scale = layer_index
            patch_sizes = c_patch_sizes[:layer_index + 1]
            out_features = c_out_features[-(layer_index + 1):]
        drop_path_rate = c_drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(c_depths))]
        drop_path = dpr[sum(c_depths[:layer_index]):sum(c_depths[:layer_index + 1])]
        if name == 'MixResViT':
            bb = MixResViT(patch_sizes=patch_sizes,
                           n_layers=c_depths[layer_index],
                           d_model=c_embed_dims[layer_index],
                           n_heads=c_num_heads[layer_index],
                           mlp_ratio=c_mlp_ratios[layer_index],
                           dropout=c_drop_rates[layer_index],
                           drop_path_rate=drop_path,
                           split_ratio=c_split_ratios[layer_index],
                           channels=in_chans,
                           n_scales=n_scales,
                           min_patch_size=min_patch_size,
                           upscale_ratio=c_upscale_ratios[layer_index],
                           out_features=out_features,
                           first_layer=first_layer,
                           layer_scale=c_layer_scale,
                           num_register_tokens=c_register_tokens)
        elif name == 'MixResNeighbour':
            bb = MixResNeighbour(patch_sizes=patch_sizes,
                                 n_layers=c_depths[layer_index],
                                 d_model=c_embed_dims[layer_index],
                                 n_heads=c_num_heads[layer_index],
                                 mlp_ratio=c_mlp_ratios[layer_index],
                                 dropout=c_drop_rates[layer_index],
                                 drop_path_rate=drop_path,
                                 attn_drop_rate=c_attn_drop_rates[layer_index],
                                 split_ratio=c_split_ratios[layer_index],
                                 channels=in_chans,
                                 cluster_size=c_cluster_sizes[layer_index],
                                 nbhd_size=c_nbhd_sizes[layer_index],
                                 n_scales=n_scales,
                                 keep_old_scale=c_keep_old_scale,
                                 scale=scale,
                                 add_image_data_to_all=c_add_image_data_to_all,
                                 min_patch_size=min_patch_size,
                                 upscale_ratio=c_upscale_ratios[layer_index],
                                 layer_scale=c_layer_scale,
                                 out_features=out_features,
                                 first_layer=first_layer)
        else:
            raise NotImplementedError(f"Unkown model: {name}")
        all_backbones.append(bb)
    model = UpDownBackbone(backbones=all_backbones,
                           backbone_dims=c_embed_dims,
                           all_out_features=c_out_features,
                           n_scales=n_scales,
                           img_size=224,
                           bb_in_feats=bb_in_feats, 
                           decoder_embed_dim=512, 
                           decoder_depth=8, 
                           decoder_num_heads=16,
                           **kwargs)
    return model


# set recommended archs
mae_mixres_small_patch32 = mae_mixres_small_patch32_dec512d8b  # decoder: 512 dim, 8 blocks