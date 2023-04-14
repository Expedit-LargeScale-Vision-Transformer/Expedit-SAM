# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock

from .image_encoder import (
    window_partition,
    window_unpartition,
    add_decomposed_rel_pos,
    ImageEncoderViT,
    Block,
    Attention,
)


class TokenClusteringBlock(nn.Module):
    def __init__(self, num_spixels=None, n_iters=5, temperture=0.01, window_size=5):
        super().__init__()
        if isinstance(num_spixels, tuple):
            assert len(num_spixels) == 2
        elif num_spixels is not None:
            x = int(math.sqrt(num_spixels))
            assert x * x == num_spixels
            num_spixels = (x, x)
        self.num_spixels = num_spixels
        self.n_iters = n_iters
        self.temperture = temperture
        assert window_size % 2 == 1
        self.r = window_size // 2

    def calc_init_centroid(self, images, num_spixels_width, num_spixels_height):
        """
        calculate initial superpixels

        Args:
            images: torch.Tensor
                A Tensor of shape (B, C, H, W)
            spixels_width: int
                initial superpixel width
            spixels_height: int
                initial superpixel height

        Return:
            centroids: torch.Tensor
                A Tensor of shape (B, C, H * W)
            init_label_map: torch.Tensor
                A Tensor of shape (B, H * W)
            num_spixels_width: int
                A number of superpixels in each column
            num_spixels_height: int
                A number of superpixels int each raw
        """
        batchsize, channels, height, width = images.shape
        device = images.device

        centroids = torch.nn.functional.adaptive_avg_pool2d(
            images, (num_spixels_height, num_spixels_width)
        )

        with torch.no_grad():
            num_spixels = num_spixels_width * num_spixels_height
            labels = (
                torch.arange(num_spixels, device=device)
                .reshape(1, 1, *centroids.shape[-2:])
                .type_as(centroids)
            )
            init_label_map = torch.nn.functional.interpolate(
                labels, size=(height, width), mode="nearest"
            ).type_as(centroids)
            init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

        init_label_map = init_label_map.reshape(batchsize, -1)
        centroids = centroids.reshape(batchsize, channels, -1)

        return centroids, init_label_map

    def forward(self, pixel_features, num_spixels=None):
        if num_spixels is None:
            num_spixels = self.num_spixels
            assert num_spixels is not None
        else:
            if isinstance(num_spixels, tuple):
                assert len(num_spixels) == 2
            else:
                x = int(math.sqrt(num_spixels))
                assert x * x == num_spixels
                num_spixels = (x, x)
        pixel_features = pixel_features.permute(0, 3, 1, 2)
        num_spixels_height, num_spixels_width = num_spixels
        num_spixels = num_spixels_width * num_spixels_height
        spixel_features, init_label_map = self.calc_init_centroid(
            pixel_features, num_spixels_width, num_spixels_height
        )

        device = init_label_map.device
        spixels_number = torch.arange(num_spixels, device=device)[None, :, None]
        relative_labels_widths = init_label_map[:, None] % num_spixels_width - spixels_number % num_spixels_width
        relative_labels_heights = torch.div(init_label_map[:, None], num_spixels_width, rounding_mode='trunc') - torch.div(spixels_number, num_spixels_width, rounding_mode='trunc')
        mask = torch.logical_and(torch.abs(relative_labels_widths) <= self.r, torch.abs(relative_labels_heights) <= self.r)
        mask_dist = (~mask) * 1e16

        pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)  # (B, C, L)
        permuted_pixel_features = pixel_features.permute(0, 2, 1)       # (B, L, C)

        for _ in range(self.n_iters):
            dist_matrix = self.pairwise_dist(pixel_features, spixel_features)    # (B, L', L)
            dist_matrix += mask_dist
            affinity_matrix = (-dist_matrix * self.temperture).softmax(1)
            spixel_features = torch.bmm(affinity_matrix.detach(), permuted_pixel_features)
            spixel_features = spixel_features / affinity_matrix.detach().sum(2, keepdim=True).clamp_(min=1e-16)
            spixel_features = spixel_features.permute(0, 2, 1)
        
        dist_matrix = self.pairwise_dist(pixel_features, spixel_features)
        hard_labels = torch.argmin(dist_matrix, dim=1)

        B, C, _ = spixel_features.shape
        spixel_features = spixel_features.permute(0, 2, 1).reshape(B, num_spixels_height, num_spixels_width, C)
        return spixel_features, hard_labels

    def pairwise_dist(self, f1, f2):
        return ((f1 * f1).sum(dim=1).unsqueeze(1)
                + (f2 * f2).sum(dim=1).unsqueeze(2)
                - 2 * torch.einsum("bcm, bcn -> bmn", f2, f1))

    def extra_repr(self):
        return f"num_spixels={self.num_spixels}, n_iters={self.n_iters}"


def naive_unpool(f_regions, region_indices):
    _, _, C = f_regions.shape
    N, L = region_indices.shape
    index = region_indices.view(N, L, 1).expand(N, L, C)
    result = f_regions.gather(1, index)
    return result


class State:
    def __init__(self, unpooling):
        self.unpooling = unpooling
        self.__updated = False

    @property
    def updated(self):
        return self.__updated

    def get(self, name, default=None):
        return getattr(self, name, default)

    def update_state(self, **states: dict):
        self.__updated = True
        for k, v in states.items():
            setattr(self, k, v)

    def call(self, input: torch.Tensor):
        return self.unpooling(input, self)


class UnpoolingBase(nn.Module):
    def forward(self, x, state: State):
        if not state.updated:
            return x, False
        return self._forward(x, state)

    def derive_unpooler(self):
        return State(self)


class NaiveUnpooling(UnpoolingBase):
    def _forward(self, x, state: State):
        return naive_unpool(x, state.hard_labels), False


class TokenReconstructionBlock(UnpoolingBase):
    def __init__(self, k=20, temperture=0.01):
        super().__init__()

        self.k = k
        self.temperture = temperture

    def _forward(self, x, state: State):
        feat = state.feat_before_pooling
        sfeat = state.feat_after_pooling
        ds = (
            (feat * feat).sum(dim=2).unsqueeze(2)
            + (sfeat * sfeat).sum(dim=2).unsqueeze(1)
            - 2 * torch.einsum("bnc, bmc -> bnm", feat, sfeat)
        )  # distance between features and super-features
        ds[ds < 0] = 0
        weight = torch.exp(-self.temperture * ds)
        if self.k >= 0:
            topk, indices = torch.topk(weight, k=self.k, dim=2)
            mink = torch.min(topk, dim=-1).values
            mink = mink.unsqueeze(-1).repeat(1, 1, weight.shape[-1])
            mask = torch.ge(weight, mink)
            zero = Variable(torch.zeros_like(weight)).to(weight.device)
            attention = torch.where(mask, weight, zero)
        attention = F.normalize(attention, dim=2)
        ret = torch.einsum("bnm, bmc -> bnc", attention, x)

        return ret, False



class HourglassImageEncoderViT(ImageEncoderViT):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        hourglass_clustering_location: int = -1,
        hourglass_num_cluster: int = 100,
        hourglass_cluster_iters: int = 5,
        hourglass_temperture: float = 0.01,
        hourglass_cluster_window_size: int = 5,
        hourglass_reconstruction_k: int = 20,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

        hourglass_clustering_location = hourglass_clustering_location if hourglass_clustering_location >= 0 else depth + 1

        self.window_size = window_size
        self.ws_new = int(math.sqrt(hourglass_num_cluster))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = HourglassBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=(window_size if i < hourglass_clustering_location else self.ws_new) if i not in global_attn_indexes else 0,
                window_size_ckpt=window_size,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.clustering_location = hourglass_clustering_location
        self.token_clustering_block = TokenClusteringBlock(
            num_spixels=hourglass_num_cluster, 
            n_iters=hourglass_cluster_iters, 
            temperture=hourglass_temperture, 
            window_size=hourglass_cluster_window_size,
        )
        self.token_reconstruction_block = TokenReconstructionBlock(
            k=hourglass_reconstruction_k,
            temperture=hourglass_temperture,
        )

    def cluster(self, x, reconstructer):
        # x: B, H, W, C
        H, W = x.shape[1:3]
        x, pad_hw = window_partition(x, self.window_size)  # B*Nw, WH, WW, C
        Bnw, _, _, C = x.shape

        reconstructer.update_state(
            feat_before_pooling=x.view(-1, self.window_size * self.window_size, C)
        )
        x, hard_labels = self.token_clustering_block(x)  # B*H*W, Wh, Ww, C
        reconstructer.update_state(hard_labels=hard_labels)
        reconstructer.update_state(feat_after_pooling=x.view(Bnw, -1, C))

        # merge window
        # Reverse window partition
        h = pad_hw[0] // self.window_size * x.shape[1]
        w = pad_hw[1] // self.window_size * x.shape[2]
        x = window_unpartition(x, self.ws_new, (h, w), (h, w))
        # out: B, h, w, C
        return x, pad_hw

    def reconstruct(self, x, H, W, recontructer, pad_hw):
        # x: B, h, w, C
        x, _ = window_partition(x, self.ws_new)   # B*Nw, Wh, Ww, C
        Bnw, _, _, C = x.shape
        x = x.view(Bnw, -1, C)

        x, _ = recontructer.call(x) # B*Nw, WH*WW, C

        # merge windows
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_unpartition(x, self.window_size, pad_hw, (H, W)) # B, H, W, C
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        H, W = x.shape[1], x.shape[2]
        reconstructer = self.token_reconstruction_block.derive_unpooler()
        reconstructer.update_state(hw_shape=(H, W))

        for i, blk in enumerate(self.blocks):
            if i == self.clustering_location:
                x, pad_hw = self.cluster(x, reconstructer)
            x = blk(x)

        if x.shape[1] != H or x.shape[2] != W:
            x = self.reconstruct(x, H, W, reconstructer, pad_hw)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

    def load_hourglass_args(self, **hourglass_args):
        hourglass_clustering_location = hourglass_args.get('hourglass_clustering_location', self.clustering_location)
        hourglass_num_cluster = hourglass_args.get('hourglass_num_cluster', self.token_clustering_block.num_spixels[0] * self.token_clustering_block.num_spixels[1])
        hourglass_cluster_iters = hourglass_args.get('hourglass_cluster_iters', self.token_clustering_block.n_iters)
        hourglass_temperture = hourglass_args.get('hourglass_temperture', self.token_clustering_block.temperture)
        hourglass_cluster_window_size = hourglass_args.get('hourglass_cluster_window_size', self.token_clustering_block.r * 2 + 1)
        hourglass_reconstruction_k = hourglass_args.get('hourglass_reconstruction_k', self.token_reconstruction_block.k)

        self.clustering_location = hourglass_clustering_location if hourglass_clustering_location >= 0 else len(self.blocks) + 1

        self.ws_new = int(math.sqrt(hourglass_num_cluster))
        for i, blk in enumerate(self.blocks):
            blk.window_size = (self.window_size if i < self.clustering_location else self.ws_new) if blk.window_size != 0 else 0
        
        self.token_clustering_block = TokenClusteringBlock(
            num_spixels=hourglass_num_cluster, 
            n_iters=hourglass_cluster_iters, 
            temperture=hourglass_temperture, 
            window_size=hourglass_cluster_window_size,
        )
        self.token_reconstruction_block = TokenReconstructionBlock(
            k=hourglass_reconstruction_k,
            temperture=hourglass_temperture,
        )


class HourglassBlock(Block):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        window_size_ckpt: int = 0,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super(HourglassBlock, self).__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            input_size=input_size,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size_ckpt, window_size_ckpt),
        )
