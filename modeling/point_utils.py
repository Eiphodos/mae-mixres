#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import math
import torch
from ..clusten import WEIGHTEDGATHERFunction


def points2img(pos, pixel, h, w):
    """
    Scatter tokens onto a canvas of size h x w
    Args:
        pos - b x n x 2, position of tokens, should be valid indices in the canvas
        pixel - b x n x c, feature of tokens
        h,w - int, height and width of the canvas
    Returns:
        img - b x c x h x w, the resulting grid img; blank spots filled with 0
    """
    b, n, c = pixel.shape
    img = torch.zeros(b, h*w, c, device=pos.device).to(pixel.dtype)
    idx = (pos[:, :, 1]*w+pos[:, :, 0]).long().unsqueeze(2).expand(-1, -1, c)  # b x n x c
    img = img.scatter(src=pixel, index=idx, dim=1)
    return img.permute(0, 2, 1).reshape(b, c, h, w)


def knn_keops(query, database, k, return_dist=False):
    """
    Compute k-nearest neighbors using the Keops library
    Backward pass turned off; Keops does not provide backward pass for distance
    Args:
        query - b x n_ x c, the position of tokens looking for knn
        database - b x n x c, the candidate tokens for knn
        k - int, the nunmber of neighbors to be found
        return_dist - bool, whether to return distance to the neighbors
    Returns:
        nn_dix - b x n x k, the indices of the knn
        nn_dist - b x n x k, if return_dist, the distance to the knn
    """
    b, n, c = database.shape
    with torch.no_grad():
        query = query.detach()
        database = database.detach()
        # Keops does not support half precision
        if query.dtype != torch.float32:
            query = query.to(torch.float32)
        if database.dtype != torch.float32:
            database = database.to(torch.float32)
        n_ = query.shape[1]
        from pykeops.torch import LazyTensor
        query_ = LazyTensor(query[:, None, :, :])
        database_ = LazyTensor(database[:, :, None, :])
        dist = ((query_-database_) ** 2).sum(-1) ** 0.5  # b x n x n_
    if return_dist:
        nn_dist, nn_idx = dist.Kmin_argKmin(k, dim=1)  # b x n_ x k
        return nn_idx, nn_dist
    else:
        nn_idx = dist.argKmin(k, dim=1)  # b x n_ x k
        return nn_idx


def shepard_decay_weights(dist, power=3):
    """
    Compute the inverse-distance weighting
    Args:
        dist - b x n x k, distances of neighbors
        power - float, the power used in inverse-distance weighting
    Returns:
        weights - b x n x k, normalized weights
    """
    dist = dist.clamp(min=1e-2)
    ipd = 1.0/(dist.pow(power)+1e-6)
    weights = ipd / (ipd.sum(dim=2, keepdim=True) + 1e-6)
    return weights


def upsample_feature_shepard(query, database, feature, database_idx=None, k=4, power=3, custom_kernel=True, nn_idx=None, return_weight_only=False):
    """
    Interpolate features in database at position in queries by interpolating knn of the positions by inverse-distance weighting
    Args:
        query - b x n x d, positions of interpolation
        database - b x n_ x d, positions of candidate knn, tokens to be interpolated
        feature - b x n_ x c, features of candidate tokens
        database_idx - b x n_ x 1, optional, indices of database tokens in the queries; if not None,
                                    replace the interpolated features with the original features in database
        k - int, number of points in neighborhood
        power - float, the power used in inverse-distance weighting
        custom_kernel - bool, whether to use custom kernel for interpolation
        nn_idx - b x n x k, optional, if not None, override k and skip knn calculation
        return_weight_only - bool, whether to return the weights of interpolation only
    Returns:
        up_features - b x n x c, interpolated features at queries
    """
    b, n_, d = database.shape
    n = query.shape[1]
    if (n == n_) and (query == database).all():
        return feature
    if nn_idx is not None:
        k = nn_idx.shape[-1]
    else:
        k = min(k, n_)
        nn_idx = knn_keops(query, database, k=k, return_dist=False)
    nn_pos = database.gather(index=nn_idx.view(b, -1, 1).expand(-1, -1, 2), dim=1).reshape(b, n, k, d)
    nn_dist = (query.unsqueeze(2) - nn_pos).pow(2).sum(-1)  # b x n x k

    nn_weights = shepard_decay_weights(nn_dist, power=power)  # b x n x k, weights of the samples
    if return_weight_only:
        return nn_weights

    c = feature.shape[-1]
    assert feature.shape[1] == n_
    if custom_kernel:
        up_features = WEIGHTEDGATHERFunction.apply(nn_idx, nn_weights, feature)
    else:
        nn_features = feature.gather(index=nn_idx.view(b, -1).unsqueeze(2).expand(-1, -1, c), dim=1).reshape(b, n, k, c)
        up_features = nn_features.mul(nn_weights.unsqueeze(3).expand(-1, -1, -1, c)).sum(dim=2)  # b x n x c

    if database_idx is not None:
         up_features.scatter_(dim=1, index=database_idx.long().expand(-1, -1, c), src=feature)
    return up_features


def find_pos_indices_in_pos(all_positions, some_positions):
    # Intended to be used to create database_idx for upsample_feature_shepard
    # Compute pairwise distances efficiently (B, N_, N)
    # Can switch from p=1 to p=2 to use euclidian distance if positions are not exactly equal.
    dists = torch.cdist(some_positions.float(), all_positions.float(), p=1)  # Manhattan distance

    # Find the index of the closest match for each element in some_positions
    pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

    return pos_indices.unsqueeze(-1)

def space_filling_cluster(pos, m, h, w, no_reorder=False, sf_type='', use_anchor=True):
    """
    The balanced clustering algorithm based on space-filling curves
    In the case where number of tokens not divisible by cluster size,
    the last cluster will have a few blank spots, indicated by the mask returned
    Args:
        pos - b x n x 2, positions of tokens
        m - int, target size of the clusters
        h,w - int, height and width
        no_reorder - bool, if True, return the clustering based on the original order of tokens;
                            otherwise, reorder the tokens so that the same cluster stays together
        sf_type - str, can be 'peano' or 'hilbert', or otherwise, horizontal scanlines w/ alternating
                        direction in each row by default
        use_anchor - bool, whether to use space-fiiling anchors or not; if False, directly compute
                            space-filling curves on the token positions
    Returns:
        pos - b x n x 2, returned only if no_reorder is False; the reordered position of tokens
        cluster_mean_pos - b x k x 2, the clustering centers
        member_idx - b x k x m, the indices of tokens in each cluster
        cluster_mask - b x k x m, the binary mask indicating the paddings in last cluster (0 if padding)
        pos_ranking - b x n x 1, returned only if no_reorder is False; i-th entry is the idx of the token
                                rank i in the new order
    """
    with torch.no_grad():
        pos = pos.detach()

        if pos.dtype != torch.float:
            pos = pos.to(torch.float)
        b, n, d = pos.shape
        if not isinstance(b, int):
            b, n, d = b.item(), n.item(), d.item()  # make the flop analyzer happy

        k = int(math.ceil(n/m))

        if not isinstance(h, int):
            h, w = h.item(), w.item()

        if use_anchor:
            patch_len = (h*w/k)**0.5
            num_patch_h = int(round(h / patch_len))
            num_patch_w = int(round(w / patch_len))
            patch_len_h, patch_len_w = h / num_patch_h, w / num_patch_w
            if sf_type == 'peano':
                num_patch_h = max(3, int(3**round(math.log(num_patch_h, 3))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 3) * (num_patch_h / 3))
                patch_len_w = w / num_patch_w
            elif sf_type == 'hilbert':
                num_patch_h = max(2, int(2**round(math.log(num_patch_h, 2))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 2) * (num_patch_h / 2))
                patch_len_w = w / num_patch_w
            hs = torch.arange(0, num_patch_h, device=pos.device)
            ws = torch.arange(0, num_patch_w, device=pos.device)
            ys, xs = torch.meshgrid(hs, ws)
            grid_pos = torch.stack([xs, ys], dim=2)  # h x w x 2
            grid_pos = grid_pos.reshape(-1, 2)

            # sort the grid centers to one line
            if sf_type == 'peano':
                order_grid_idx, order_idx = calculate_peano_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            elif sf_type == 'hilbert':
                order_grid_idx, order_idx = calculate_hilbert_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            else:
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys*w
                order_mask[1::2] += (w-1)
                order_mask = order_mask.reshape(-1)
                order_idx = order_mask.sort()[1]
                order_idx_src = torch.arange(len(order_idx)).to(pos.device)
                order_grid_idx = torch.zeros_like(order_idx_src)
                order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)

            ordered_grid = grid_pos[order_idx]
            patch_len_hw = torch.Tensor([patch_len_w, patch_len_h]).to(pos.device)

            init_pos_means = ordered_grid * patch_len_hw + patch_len_hw/2 - 0.5
            nump = ordered_grid.shape[0]

            prev_means = torch.zeros_like(init_pos_means)
            prev_means[1:] = init_pos_means[:nump-1].clone()
            prev_means[0] = prev_means[1] - (prev_means[2]-prev_means[1])  # float('inf')
            next_means = torch.zeros_like(init_pos_means)
            next_means[:nump-1] = init_pos_means[1:].clone()
            next_means[-1] = next_means[-2] + (next_means[-2]-next_means[-3])  # float('inf')

            mean_assignment = (pos / patch_len_hw).floor()
            mean_assignment = mean_assignment[..., 0] + mean_assignment[..., 1] * num_patch_w
            mean_assignment = order_grid_idx.unsqueeze(0).expand(b, -1).gather(index=mean_assignment.long(), dim=1).unsqueeze(2)  # b x n x 1

            prev_mean_assign = prev_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d), dim=1)  # b x n x d
            next_mean_assign = next_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d), dim=1)  # b x n x d
            dist_prev = (pos-prev_mean_assign).pow(2).sum(-1)  # b x n
            dist_next = (pos-next_mean_assign).pow(2).sum(-1)
            dist_ratio = dist_prev / (dist_next + 1e-5)

            pos_ranking = mean_assignment * (dist_ratio.max()+1) + dist_ratio.unsqueeze(2)
            pos_ranking = pos_ranking.sort(dim=1)[1]  # b x n x 1

        else:
            if sf_type == 'peano':
                _, pos_ranking = calculate_peano_order(h, w, pos)
            elif sf_type == 'hilbert':
                _, pos_ranking = calculate_hilbert_order(h, w, pos)
            else:
                hs = torch.arange(0, h, device=pos.device)
                ws = torch.arange(0, w, device=pos.device)
                ys, xs = torch.meshgrid(hs, ws)
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys*w
                order_mask[1::2] += (w-1)
                order_mask = order_mask.reshape(-1)
                pos_idx = pos[..., 0] + pos[..., 1] * w
                order_mask = order_mask.gather(index=pos_idx.long().reshape(-1), dim=0).reshape(b, n)
                pos_ranking = order_mask.sort()[1]
            pos_ranking = pos_ranking.unsqueeze(2)

        pos = pos.gather(index=pos_ranking.expand(-1, -1, d), dim=1)  # b x n x d

        if k*m == n:
            cluster_mask = None
            cluster_mean_pos = pos.reshape(b, k, -1, d).mean(2)
        else:
            pos_pad = torch.zeros(b, k*m, d, dtype=pos.dtype, device=pos.device)
            pos_pad[:, :n] = pos.clone()
            cluster_mask = torch.zeros(b, k*m, device=pos.device).long()
            cluster_mask[:, :n] = 1
            cluster_mask = cluster_mask.reshape(b, k, m)
            cluster_mean_pos = pos_pad.reshape(b, k, -1, d).sum(2) / cluster_mask.sum(2, keepdim=True)

        if no_reorder:
            if k*m == n:
                member_idx = pos_ranking.reshape(b, k, m)
            else:
                member_idx = torch.zeros(b, k*m, device=pos.device, dtype=torch.int64)
                member_idx[:, :n] = pos_ranking.squeeze(2)
                member_idx = member_idx.reshape(b, k, m)
            return cluster_mean_pos, member_idx, cluster_mask
        else:
            member_idx = torch.arange(k*m, device=pos.device)
            member_idx[n:] = 0
            member_idx = member_idx.unsqueeze(0).expand(b, -1)  # b x k*m
            member_idx = member_idx.reshape(b, k, m)

            return pos, cluster_mean_pos, member_idx, cluster_mask, pos_ranking


def upsample_shepard_cdist(
        query,
        database,
        feature,
        eps=1e-9
):
    """
    Vectorized approach that:
      1) Finds which queries exactly/near-exactly match a database position
      2) *Skips* interpolation for those exact queries
      3) Interpolates only the non-exact queries in a single batched call
      4) Reintegrates results so final_features[b,iQ] is either:
         - The original feature if query[b,iQ] is exact
         - The interpolated feature otherwise
    Assumes each batch has the same number 'E' of exact matches.

    Args:
      query:    (B, nQ, D) positions to upsample
      database: (B, nDB, D) known positions
      feature:  (B, nDB, C) features at those positions
      upsample_fn: function that does interpolation:
                   upsample_fn(query_sub, database_sub, feature_sub) -> (B, nSub, C)
      eps:      threshold for "exact" (floating coords). Use eps=0 if integer.

    Returns:
      final_features: (B, nQ, C)
    """
    device = query.device
    B, nQ, D = query.shape
    B2, nDB, D2 = database.shape
    B3, nDB2, C = feature.shape
    assert B == B2 == B3, "Batch dimension mismatch"
    assert D == D2, "Position dimension mismatch"
    assert nDB == nDB2, "Database vs. feature mismatch"

    # ------------------------------------------------------------
    # 1) Identify which queries are "exact" matches.
    #    We'll say min distance < eps => "exact match"
    #    cdist => shape [B, nQ, nDB]
    # ------------------------------------------------------------
    dists = torch.cdist(query, database)  # [B, nQ, nDB]
    min_dists, min_idxs = dists.min(dim=2)  # both => [B, nQ]
    exact_mask = (min_dists < eps)  # bool [B, nQ]

    # Verify each batch has the same number E of exact matches
    # (User has told us that is guaranteed, but let's check for safety.)
    exact_counts = exact_mask.sum(dim=1)  # [B], number of exact matches per batch
    E = exact_counts[0].item()
    if not bool((exact_counts == E).all()):
        raise ValueError("Not all batches have the same # of exact matches, but it was assumed!")
    # So each batch has exactly E exact queries => nQ-E non-exact.

    # ------------------------------------------------------------
    # 2) We'll reorder each batch so that the "non-exact" queries come first
    #    Then we can easily slice out the first (nQ - E) as the sub-tensor.
    #
    #    Trick:
    #      - Make an index array [0..nQ-1]
    #      - Sort it so that the "non-exact" queries come first,
    #        preserving their original order among themselves (stable sort).
    # ------------------------------------------------------------
    idx_arange = torch.arange(nQ, device=device).unsqueeze(0).expand(B, nQ)  # (B, nQ)
    # Convert exact_mask to an int, so "non-exact" = 1, "exact" = 0
    # and use that as a high-level sorting key.
    # We want the non-exact queries to appear first => bigger "sort key".
    sort_key = (~exact_mask).to(torch.int64)  # shape (B, nQ), has 1 for non-exact, 0 for exact

    # Combine with original index to keep stable ordering among queries that share the same sort_key:
    # We'll multiply the primary key by a large enough factor to ensure it dominates,
    # then add the original index as a tiebreaker for stability.
    # e.g. if nQ <= 65535, we can do:
    #   combined_key = sort_key * 100000 + idx_arange
    # or simply something bigger than nQ:
    combined_key = sort_key * (nQ + 1) + idx_arange

    # Sort ascending => those with bigger combined_key come last,
    # but we want non-exact (sort_key=1) first.
    # So we can either invert the logic or simply sort descending:
    # If you do descending => non-exact (sort_key=1) get bigger keys than exact (sort_key=0),
    # so they'll come first.
    sorted_indices = combined_key.argsort(dim=1, descending=True)  # (B, nQ)

    # The first (nQ - E) entries in `sorted_indices` are the non‐exact queries for each batch
    # The last E entries are the exact queries for each batch
    nonexact_count = nQ - E

    idx_nonexact = sorted_indices[:, :nonexact_count]  # (B, nQ-E)
    idx_exact = sorted_indices[:, nonexact_count:]  # (B, E)

    # ------------------------------------------------------------
    # 3) Gather the non-exact queries => shape (B, nQ-E, D)
    #    Then do one big interpolation call.
    # ------------------------------------------------------------
    # Expand for gather:
    gather_idx_nonexact = idx_nonexact.unsqueeze(-1).expand(-1, -1, D)  # (B, nQ-E, D)
    # Gather from 'query'
    query_nonexact = torch.gather(query, dim=1, index=gather_idx_nonexact)

    # We do the same for 'database' and 'feature' if your upsample_fn expects (B, nDB, D)/(B, nDB, C) for each batch.
    # Typically your upsample_fn can just see the full database. So:
    #   upsample_fn(query_nonexact, database, feature) -> (B, nQ-E, C)
    # If your upsample_fn uses KNN, it will internally do partial computations only for these queries.

    up_features_nonexact = upsample_feature_shepard(query_nonexact, database, feature)  # (B, nQ-E, C)

    # ------------------------------------------------------------
    # 4) Build the final output of shape (B, nQ, C).
    #    We'll fill in:
    #      - the non-exacts from up_features_nonexact
    #      - the exact matches from 'feature' at the right db index
    # ------------------------------------------------------------
    final_features = torch.empty(B, nQ, C, device=device, dtype=feature.dtype)

    # (a) Scatter the interpolation results for non‐exact queries
    # We want final_features[b, idx_nonexact[b], :] = up_features_nonexact[b, :, :]
    gather_idx_nonexact_c = idx_nonexact.unsqueeze(-1).expand(-1, -1, C)  # (B, nQ-E, C)
    final_features.scatter_(dim=1, index=gather_idx_nonexact_c, src=up_features_nonexact)

    # (b) For exact queries, copy from the original 'feature' using min_idxs
    #     Because for an exact query iQ, min_idxs[b,iQ] = that DB index
    #     We'll gather the matching features => shape (B, nQ, C),
    #     then scatter that into final_features only at the exact spots.
    #
    # We'll do it in a single advanced-index step.
    # First gather from feature => shape (B, nQ, C)
    # We can gather all at once, then we only scatter the exact ones.
    # But let's do it more directly:

    # Build a flat view:
    b_arange = torch.arange(B, device=device).unsqueeze(1).expand(B, nQ)  # (B, nQ)
    # Flatten everything to 1D to do advanced indexing
    b_flat = b_arange.reshape(-1)  # (B*nQ)
    iQ_flat = torch.arange(nQ, device=device).unsqueeze(0).expand(B, nQ).reshape(-1)  # (B*nQ)
    db_flat = min_idxs.reshape(-1)  # (B*nQ)
    mask_flat = exact_mask.reshape(-1)  # bool (B*nQ)

    # We can gather all features in one shot => feature[b_flat, db_flat], shape (B*nQ, C),
    # then apply the mask.
    matched_feats = feature[b_flat, db_flat]  # (B*nQ, C)
    # matched_feats[mask_flat] are the features for the "exact" queries.

    # Now place them into final_features in a single shot.
    # final_features[b_flat[mask_flat], iQ_flat[mask_flat]] = matched_feats[mask_flat]
    # We'll do that with advanced indexing:
    final_features[b_flat[mask_flat], iQ_flat[mask_flat]] = matched_feats[mask_flat]

    return final_features



def calculate_peano_order(h, w, pos):
    """
    Given height and width of the canvas and position of tokens,
    calculate the peano curve order of the tokens
    Args:
        h,w - int, height and width
        pos - b x n x 2, positions of tokens
    Returns:
        final_order_ - b x n, i-th entry is the rank of i-th token in the new order
        final_order_index - b x n, i-th entry is the idx of the token rank i in the new order
    """
    b, n, _ = pos.shape
    num_levels = math.ceil(math.log(h, 3))
    assert num_levels >= 1, "h too short"
    first_w = None
    if h != w:
        first_w = round(3 * (w/h))
        if first_w == 3:
            first_w = None
    init_dict = torch.Tensor([[2, 3, 8], [1, 4, 7], [0, 5, 6]]).to(pos.device)
    inverse_dict = torch.Tensor([[[1, 1], [1, -1], [1, 1]], [[-1, 1], [-1, -1], [-1, 1]], [[1, 1], [1, -1], [1, 1]]]).to(pos.device)
    if first_w is not None:
        init_dict_flip = init_dict.flip(dims=[0])
        init_dict_f = torch.cat([init_dict, init_dict_flip], dim=1)  # 3 x 6
        init_dict_f = init_dict_f.repeat(1, math.ceil(first_w/6))
        init_dict_f = init_dict_f[:, :first_w]  # 3 x fw
        w_index = torch.arange(math.ceil(first_w/3)).to(pos.device).repeat_interleave(3)[:first_w] * 9  # fw
        init_dict_f = init_dict_f + w_index
        init_dict_f = init_dict_f.reshape(-1)  # 3*fw
        inverse_dict_f = inverse_dict[:, :2].repeat(1, math.ceil(first_w/2), 1)[:, :first_w]  # 3 x fw x 2
        inverse_dict_f = inverse_dict_f.reshape(-1, 2)
    init_dict = init_dict.reshape(-1)  # 9
    inverse_dict = inverse_dict.reshape(-1, 2)  # 9 x 2
    last_h = h
    rem_pos = pos
    levels_pos = []
    for le in range(num_levels):
        cur_h = last_h / 3
        level_pos = (rem_pos / cur_h).floor()
        levels_pos.append(level_pos)
        rem_pos = rem_pos % cur_h
        last_h = cur_h
    orders = []
    for i in range(len(levels_pos)):
        inverse = torch.ones_like(pos)  # b x n x 2
        for j in range(i):
            cur_level_pos = levels_pos[i-j-1]
            if i-j-1 == 0 and first_w is not None:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * first_w  # b x n
                cur_inverse = inverse_dict_f.gather(index=cur_level_pos_index.long().view(-1, 1).expand(-1, 2), dim=0).reshape(b, n, 2)
            else:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * 3  # b x n
                cur_inverse = inverse_dict.gather(index=cur_level_pos_index.long().view(-1, 1).expand(-1, 2), dim=0).reshape(b, n, 2)
            inverse = cur_inverse * inverse
        level_pos = levels_pos[i]
        inversed_pos = torch.where(inverse > 0, level_pos, 2-level_pos)
        if i == 0 and first_w is not None:
            inversed_pos_index = inversed_pos[..., 0] + inversed_pos[..., 1] * first_w  # b x n
            cur_order = init_dict_f.gather(index=inversed_pos_index.long().view(-1), dim=0).reshape(b, n)
        else:
            inversed_pos_index = inversed_pos[..., 0] + inversed_pos[..., 1] * 3  # b x n
            cur_order = init_dict.gather(index=inversed_pos_index.long().view(-1), dim=0).reshape(b, n)
        orders.append(cur_order)
    final_order = orders[-1]
    for i in range(len(orders)-1):
        cur_order = orders[i]
        final_order = final_order + cur_order * (9**(num_levels-i-1))
    final_order_index = final_order.sort(dim=1)[1]
    order_src = torch.arange(n).to(pos.device).unsqueeze(0).expand(b, -1)  # b x n
    final_order_ = torch.zeros_like(order_src)
    final_order_.scatter_(index=final_order_index, src=order_src, dim=1)
    return final_order_, final_order_index


def calculate_hilbert_order(h, w, pos):
    """
    Given height and width of the canvas and position of tokens,
    calculate the hilber curve order of the tokens
    Args:
        h,w - int, height and width
        pos - b x n x 2, positions of tokens
    Returns:
        final_order_ - b x n, i-th entry is the rank of i-th token in the new order
        final_order_index - b x n, i-th entry is the idx of the token rank i in the new order
    """
    b, n, _ = pos.shape
    num_levels = math.ceil(math.log(h, 2))
    assert num_levels >= 1, "h too short"
    first_w = None
    if h != w:
        first_w = round(2 * (w/h))
        if first_w == 2:
            first_w = None
    rotate_dict = torch.Tensor([[[-1, 1], [0, 0]], [[0, -1], [0, 1]], [[1, 0], [-1, 0]]]).to(pos.device)  # 3 x 2 x 2 -1 means left, 1 means right
    if first_w is not None:
        rotate_dict_f = rotate_dict[0].repeat(1, math.ceil(first_w/2))[:, :first_w]  # 2 x fw
        rotate_dict_f = rotate_dict_f.reshape(-1)  # 2*fw
    rotate_dict = rotate_dict.reshape(3, -1)  # 3 x 4
    rot_res_dict = torch.Tensor([[0, 3, 1, 2], [2, 3, 1, 0], [2, 1, 3, 0], [0, 1, 3, 2]]).to(pos.device)  # 4 x 4
    last_h = h
    rem_pos = pos
    levels_pos = []
    for le in range(num_levels):
        cur_h = last_h / 2
        level_pos = (rem_pos / cur_h).floor()
        levels_pos.append(level_pos)
        rem_pos = rem_pos % cur_h
        last_h = cur_h
    orders = []
    for i in range(len(levels_pos)):
        level_pos = levels_pos[i]
        if i == 0 and first_w is not None:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * first_w  # b x n
        else:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * 2  # b x n
        rotate = torch.zeros_like(pos[..., 0])
        for j in range(i):
            cur_level_pos = levels_pos[j]
            if j == 0 and first_w is not None:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * first_w  # b x n
                cur_rotate = rotate_dict_f.gather(index=cur_level_pos_index.long().view(-1), dim=0).reshape(b, n)
            else:
                rotate_d = rotate_dict.gather(index=(rotate % 3).long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n, 4)
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * 2  # b x n
                cur_rotate = rotate_d.gather(index=cur_level_pos_index.long().unsqueeze(2), dim=2).reshape(b, n)
            rotate = cur_rotate + rotate
        rotate = rotate % 4
        rotate_res = rot_res_dict.gather(index=rotate.long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n, 4)
        rotate_res = rotate_res.gather(index=level_pos_index.long().unsqueeze(2), dim=2).squeeze(2)  # b x n
        orders.append(rotate_res)
    final_order = orders[-1]
    for i in range(len(orders)-1):
        cur_order = orders[i]
        final_order = final_order + cur_order * (4**(num_levels-i-1))
    final_order_index = final_order.sort(dim=1)[1]
    order_src = torch.arange(n).to(pos.device).unsqueeze(0).expand(b, -1)  # b x n
    final_order_ = torch.zeros_like(order_src)
    final_order_.scatter_(index=final_order_index, src=order_src, dim=1)
    return final_order_, final_order_index


def upsample_by_patch(features, positions, tokens_per_scale):
    B, N, C = features.shape
    device = features.device
    n_scales = len(tokens_per_scale)
    ps = [2 ** (n_scales - s - 1) for s in range(n_scales)]
    start_id = 0
    scale_blocks = {}
    for s, t, p in zip(range(n_scales), tokens_per_scale, ps):
        end_id = start_id + t
        scale_blocks[s] = (start_id, end_id, p)
        start_id = end_id
    all_new_feats = []
    all_new_pos = []
    for scale, (start, end, patch_size) in scale_blocks.items():
        feat_s = features[:, start:end, :]
        pos_s = positions[:, start:end, :]
        B_s, Ns, _ = pos_s.shape
        offsets = torch.arange(patch_size, device=device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')
        dxy = torch.stack([dx, dy], dim=-1).reshape(-1, 2)
        pos_s_exp = pos_s.unsqueeze(2) + dxy.view(1, 1, -1, 2)
        pos_s_exp = pos_s_exp.view(B, -1, 2)
        feat_s_exp = feat_s.unsqueeze(2).repeat(1, 1, patch_size**2, 1).view(B, -1, C)
        all_new_feats.append(feat_s_exp)
        all_new_pos.append(pos_s_exp)
    final_feats = torch.cat(all_new_feats, dim=1)
    final_pos = torch.cat(all_new_pos, dim=1)
    return final_feats, final_pos


def hierarchical_upsample_ordered(features, positions, tokens_per_scale, input_shape):
    B, N, C = features.shape
    device = features.device
    H, W = input_shape
    visibility = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    n_scales = len(tokens_per_scale)
    ps = [2 ** (n_scales - s - 1) for s in range(n_scales)]
    start_id = 0
    scale_blocks = []
    for t, p in zip(tokens_per_scale, ps):
        end_id = start_id + t
        scale_blocks.append((start_id, end_id, p))
        start_id = end_id
    scale_blocks = scale_blocks[::-1]
    all_feats = []
    all_pos = []
    for start, end, patch_size in scale_blocks:
        feats_s = features[:, start:end, :]        # (B, Ns, C)
        pos_s = positions[:, start:end, :]         # (B, Ns, 2)
        B_s, Ns, _ = pos_s.shape
        dx, dy = torch.meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device), indexing='ij')
        offset = torch.stack([dx, dy], dim=-1).reshape(-1, 2)  # (ps², 2)
        pos_exp = pos_s.unsqueeze(2) + offset.view(1, 1, -1, 2)  # (B, Ns, ps², 2)
        pos_exp = pos_exp.view(B, -1, 2).long()
        xg = pos_exp[:, :, 0]
        yg = pos_exp[:, :, 1]
        flat_visibility = visibility.view(B, -1)  # (B, H*W)
        idx_flat = yg * W + xg
        idx_batch = torch.arange(B, device=device).view(B, 1).repeat(1, idx_flat.shape[1]).long()
        claimed = flat_visibility[idx_batch, idx_flat].view(B, Ns, patch_size**2).any(dim=2)  # (B, Ns)
        keep = ~claimed
        if keep.sum() == 0:
            continue
        B_idx, Ns_idx = torch.nonzero(keep, as_tuple=True)
        pos_keep = pos_s[B_idx, Ns_idx]
        feat_keep = feats_s[B_idx, Ns_idx]
        pos_keep = pos_keep.view(B, -1, 2)
        feat_keep = feat_keep.view(B, -1, C)
        pos_grid = pos_keep.unsqueeze(2) + offset.view(1, 1, -1, 2)  # (N_keep, ps², 2)
        pos_grid = pos_grid.view(B, -1, 2).long()
        feat_grid = feat_keep.unsqueeze(2).repeat(1, 1, patch_size**2, 1).view(B, -1, C)
        all_feats.append(feat_grid)
        all_pos.append(pos_grid)
        x_vis = pos_grid[:, :, 0]
        y_vis = pos_grid[:, :, 1]
        b_vis = torch.arange(B).unsqueeze(-1).expand(-1, pos_grid.shape[1])
        visibility[b_vis, y_vis, x_vis] = True
    return torch.cat(all_feats, dim=1), torch.cat(all_pos, dim=1)


def upsample_tokens_fixed_scales(features, positions, tokens_per_scale):
    B, N, C = features.shape
    device = features.device
    n_scales = len(tokens_per_scale)
    ps = [2 ** (n_scales - s - 1) for s in range(n_scales)]
    start_id = 0
    scale_blocks = []
    for s, t, p in zip(range(n_scales), tokens_per_scale, ps):
        end_id = start_id + t
        scale_blocks[s] = (start_id, end_id, p)
        start_id = end_id

    all_new_feats = []
    all_new_pos = []

    for scale, (start, end, patch_size) in scale_blocks.items():
        feat_s = features[:, start:end, :]              # (B, Ns, C)
        pos_s = positions[:, start:end, :]             # (B, Ns, 2) — just x, y

        B_s, Ns, _ = pos_s.shape

        # Generate relative grid positions (ps x ps x 2)
        offsets = torch.arange(patch_size, device=device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')
        dxy = torch.stack([dx, dy], dim=-1).reshape(-1, 2)  # (ps*ps, 2)

        # Expand pos_s: (B, Ns, 1, 2) + (1, 1, ps*ps, 2)
        pos_s_exp = pos_s.unsqueeze(2) + dxy.view(1, 1, -1, 2)  # (B, Ns, ps*ps, 2)
        pos_s_exp = pos_s_exp.view(B, -1, 2)                    # (B, Ns * ps^2, 2)

        # Expand feats: repeat each feature ps^2 times
        feat_s_exp = feat_s.unsqueeze(2).repeat(1, 1, patch_size**2, 1).view(B, -1, C)  # (B, Ns * ps^2, C)

        all_new_feats.append(feat_s_exp)
        all_new_pos.append(pos_s_exp)

    # Concatenate across all scales
    final_feats = torch.cat(all_new_feats, dim=1)  # (B, N', C)
    final_pos = torch.cat(all_new_pos, dim=1)      # (B, N', 3)

    return final_feats, final_pos
