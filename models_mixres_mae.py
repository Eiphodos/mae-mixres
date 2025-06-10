# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

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
    def __init__(self, backbones, backbone_dims, out_dim, all_out_features, n_scales, num_classes, bb_in_feats):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        final_upsampling_ratios = []
        for b in self.backbones:
            final_upsampling_ratios.append(b.upscale_ratio)
        self.final_upsampling_ratios = final_upsampling_ratios
        self.out_dim = out_dim
        self.backbone_dims = backbone_dims
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}
        self.n_scales = n_scales
        self.bb_in_feats = bb_in_feats
        scales = list(range(self.n_scales))
        self.bb_scales = scales + scales[-2::-1]
        self.num_classes = num_classes

        tot_out_dim = backbone_dims[-1]
        #self.head_norm = nn.LayerNorm(tot_out_dim)
        #self.head = MLP(tot_out_dim, tot_out_dim, num_classes, num_layers=3)
        self.head = nn.Linear(tot_out_dim, num_classes)

        self.apply(self._init_weights)

        print("Successfully built UpDownBackbone model!")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, im):
        B, C, H, W = im.shape
        up = True
        upsampling_mask = None
        features = None
        features_pos = None
        outs = {}
        for j in range(len(self.backbones)):
            scale = self.bb_scales[j]
            output = self.backbones[j](im, scale, features, features_pos, upsampling_mask)
            bb_out_features = self.backbones[j]._out_features
            all_feat = []
            all_scale = []
            all_pos = []
            all_ss = []
            for i, f in enumerate(bb_out_features):
                feat = output[f]
                feat_pos = output[f + '_pos']
                feat_scale = output[f + '_scale']
                feat_ss = output[f + '_spatial_shape']
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
                upsampling_mask = self.generate_random_upsampling_mask(B, N)
                #upsampling_mask = self.upsamplers[scale](all_feat[0]).squeeze(-1)

            #print("Upsampling mask for scale {}: pred: {}, oracle: {}".format(scale, upsampling_mask_pred.shape, upsampling_mask_oracle.shape))

            if j < len(self.backbones) - 1:
                all_pos = torch.cat(all_pos, dim=1)
                all_scale = torch.cat(all_scale, dim=1)
                features_pos = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
                features = torch.cat(all_feat, dim=1)
                #print("For bb level {}, feature shape is {}".format(j, features.shape))

        outs['min_spatial_shape'] = output['min_spatial_shape']

        grid_pos = get_2dpos_of_curr_ps_in_min_ps(H, W, self.backbones[-1].patch_size, self.backbones[-1].min_patch_size, B)
        mask, ids_restore = self.get_mask_and_ids_restore(grid_pos, output[self.all_out_features[-1] + '_pos'])
        return output[self.all_out_features[-1]], mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


    def generate_random_upsampling_mask(self, batch_size, n_tokens):
        upsampling_mask = torch.randn(batch_size, n_tokens).float().to('cuda')
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


def get_2dpos_of_curr_ps_in_min_ps(height, width, patch_size, min_patch_size, batch_size):
    patches_coords = torch.meshgrid(torch.arange(0, width // min_patch_size, patch_size // min_patch_size), torch.arange(0, height // min_patch_size, patch_size // min_patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.transpose(0, 1)
    patches_coords = patches_coords.reshape(-1, 2)
    patches_coords = patches_coords.repeat(batch_size, 1, 1)

    return patches_coords