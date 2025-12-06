# Inspired by timm and swin paper

import math
import collections.abc
from typing import Optional, Tuple
from collections import OrderedDict
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

# add timm import (optional)
try:
    import timm
except Exception:
    timm = None


def to_2tuple(x):
    """Convert int or 1/2-length iterable to a 2-tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        x_list = list(x)
        if len(x_list) == 2:
            return (x_list[0], x_list[1])
        if len(x_list) == 1:
            return (x_list[0], x_list[0])
        raise ValueError("Iterable must have 1 or 2 elements.")
    if isinstance(x, (int, float)):
        return (x, x)
    raise TypeError(f"Unsupported type for to_2tuple: {type(x)}")


def trunc_normal_(tensor, mean=0.0, std=1.0):
    """Normal init as a lightweight stand-in."""
    with torch.no_grad():
        return tensor.normal_(mean, std)


class DropPath(nn.Module):
    """Stochastic Depth."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


class Mlp(nn.Module):
    """2-layer MLP with dropout and activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):
    """Split (B,H,W,C) into windows of size (window_size, window_size)."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad W then H
        H_padded, W_padded = H + pad_h, W + pad_w
    else:
        H_padded, W_padded = H, W

    x = x.reshape(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Reverse window_partition back to (B,H,W,C)."""
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_padded, W_padded = H + pad_h, W + pad_w

    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    x = windows.reshape(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H_padded, W_padded, -1)

    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :]
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: (num_windows*B, N, C), N = window_size*window_size
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].reshape(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + rel_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Standard Swin block with optional shift."""
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(window_size), num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None, attn_mask=None):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        H = H if H is not None else int(math.sqrt(L))
        W = W if W is not None else int(math.sqrt(L))
        assert L == H * W, "Wrong token length."

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # build mask for shifted attention
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # pad to window grid
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        shifted_x = F.pad(shifted_x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = shifted_x.shape

        # window attention
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # reverse windows and shift
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # crop padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample by 2x (spatial)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """Stack of Swin/Factorized blocks + optional downsample."""
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        shift_size=0,
        use_checkpoint=False,
        use_factorized_attention=False,
        spectral_size=1,
        spatial_shift_size=0,
        act_layer=nn.GELU,
        spectral_option=1,
    ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = shift_size
        self.downsample = downsample

        self.blocks = nn.ModuleList()
        dpr = drop_path if isinstance(drop_path, list) else [drop_path] * depth

        for i in range(depth):
            if use_factorized_attention:
                block_type = i % 3  # spatial -> spatial_shifted -> spectral
                use_spectral = (block_type == 2)
                current_shift_size = spatial_shift_size if block_type == 1 else 0
                block = FactorizedSwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    spectral_size=spectral_size,
                    window_size=window_size,
                    shift_size=current_shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_spectral=use_spectral,
                    spectral_option=spectral_option,
                )
            else:
                current_shift_size = 0 if (i % 2 == 0) else self.shift_size
                block = SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=current_shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            self.blocks.append(block)

    def forward(self, x, H, W):
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        for blk in self.blocks:
            if getattr(self, "use_checkpoint", False):
                x = torch.utils.checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        return x, H, W


class PatchEmbed(nn.Module):
    """Patch embedding for RGB."""
    def __init__(self, img_size, patch_size, embed_dim, in_chans=3, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone (RGB)."""
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        downsamples = []
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            downsamples.append(PatchMerging(dim=layer_dim, norm_layer=norm_layer) if i_layer < self.num_layers - 1 else None)

        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            self.layers.append(
                BasicLayer(
                    dim=layer_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=downsamples[i_layer - 1],
                    shift_size=window_size // 2,
                )
            )

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, L, C_out)"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        H_p, W_p = self.patch_embed.patches_resolution
        for layer in self.layers:
            x, H_p, W_p = layer(x, H_p, W_p)

        x = self.norm(x)
        return x


class HSIPatchMerging(nn.Module):
    """Downsample by 2x (spatial) while preserving spectral grouping."""
    def __init__(self, dim, norm_layer, spectral_size=1):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.spectral_size = spectral_size

    def forward(self, x, H, W):
        """
        x: (B, S*H*W, C), returns (B, S*(H/2)*(W/2), 2*C)
        """
        B, L, C = x.shape
        assert L == H * W * self.spectral_size, "Wrong token length."
        x = x.view(B, self.spectral_size, H, W, C)

        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H_padded, W_padded = H + (H % 2), W + (W % 2)
        else:
            H_padded, W_padded = H, W

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, self.spectral_size * (H_padded // 2) * (W_padded // 2), 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SpectralConv(nn.Module):
    """Global SxS mixing along spectral axis (shared across channels)."""
    def __init__(self, spectral_size, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.spectral_size = spectral_size
        self.spec_weight = nn.Parameter(torch.empty(spectral_size, spectral_size))
        self.spec_bias = nn.Parameter(torch.zeros(spectral_size))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spec_weight)
        nn.init.zeros_(self.spec_bias)

    def forward(self, x):
        # x: (B*H*W, S, C)
        B_, S, C = x.shape
        assert S == self.spectral_size, f"Expected S={self.spectral_size}, got {S}"
        x_cs = x.transpose(1, 2)  # (B_, C, S)
        y_cs = F.linear(x_cs, self.spec_weight, self.spec_bias)
        y = y_cs.transpose(1, 2)  # (B_, S, C)
        y = self.attn_drop(y)
        y = self.proj_drop(y)
        return y


class FactorizedSwinBlock(nn.Module):
    """Spatial window attention or global spectral op + MLP."""
    def __init__(
        self,
        dim,
        num_heads,
        spectral_size,
        window_size,
        shift_size,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_spectral=True,
        spectral_option=1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.spectral_size = spectral_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_spectral = use_spectral

        self.norm1 = norm_layer(dim)

        if use_spectral:
            if spectral_option == 1:
                self.spectral_module = SpectralGlobalAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    spectral_size=self.spectral_size,
                )
            elif spectral_option == 2:
                self.spectral_module = SpectralConv(spectral_size=self.spectral_size, attn_drop=attn_drop, proj_drop=drop)
            else:
                raise ValueError(f"Unknown spectral_option: {spectral_option}")
        else:
            self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self._cached_masks = {}

    def _get_cached_mask(self, H, W):
        key = (H, W, self.window_size, self.shift_size)
        if key not in self._cached_masks:
            self._cached_masks[key] = self._calculate_mask(H, W, self.window_size, self.shift_size)
        return self._cached_masks[key]

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        spectral_size = self.spectral_size
        spatial_size = L // spectral_size
        assert L == spectral_size * spatial_size, "Wrong token length."

        H = H if H is not None else int(math.sqrt(spatial_size))
        W = W if W is not None else int(math.sqrt(spatial_size))

        shortcut = x
        x = self.norm1(x)

        if self.use_spectral:
            # global operation along spectral axis
            x = x.reshape(B, spectral_size, H * W, C).permute(0, 2, 1, 3).reshape(B * H * W, spectral_size, C)
            x = self.spectral_module(x)
            x = x.reshape(B, H * W, spectral_size, C).permute(0, 2, 1, 3).reshape(B, spectral_size * H * W, C)
        else:
            x = self._apply_spatial_attention(x, B, spectral_size, H, W, self.shift_size)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _apply_spatial_attention(self, x_spatial, B, spectral_size, H, W, shift_size):
        # tokens -> (B*S, H, W, C)
        x_spatial = x_spatial.view(B * spectral_size, H, W, self.dim)

        if shift_size > 0:
            x_spatial = torch.roll(x_spatial, shifts=(-shift_size, -shift_size), dims=(1, 2))

        windows = window_partition(x_spatial, self.window_size).view(-1, self.window_size * self.window_size, self.dim)
        attn_mask = self._get_cached_mask(H, W) if shift_size > 0 else None
        attn_windows = self.attn(windows, mask=attn_mask)

        # windows -> tokens
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x_spatial = window_reverse(attn_windows, self.window_size, H, W)

        if shift_size > 0:
            x_spatial = torch.roll(x_spatial, shifts=(shift_size, shift_size), dims=(1, 2))

        return x_spatial.reshape(B, spectral_size * H * W, self.dim)

    def _calculate_mask(self, H, W, window_size, shift_size):
        """Attention mask for shifted windows."""
        img_mask = torch.zeros((1, H, W, 1), device=self.attn.relative_position_bias_table.device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size).reshape(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class AdaptiveHSIPatchEmbed(nn.Module):
    """Patch embedding for HSI with grouping options."""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, spectral_group_size=None, norm_layer=nn.LayerNorm, embedding_option=1):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.embed_dim = embed_dim
        self.embedding_option = embedding_option

        if embedding_option == 1:
            self.spectral_size = (in_chans + spectral_group_size - 1) // spectral_group_size
            self.proj = nn.Conv2d(in_chans, embed_dim * self.spectral_size, kernel_size=patch_size, stride=patch_size)
        elif embedding_option == 2:
            self.spectral_size = (in_chans + spectral_group_size - 1) // spectral_group_size
            if self.spectral_size > 1:
                self.proj = nn.Conv2d(spectral_group_size, embed_dim, kernel_size=patch_size, stride=patch_size)
            else:
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif embedding_option == 3:
            self.spectral_size = (in_chans + spectral_group_size - 1) // spectral_group_size
            self.proj = nn.Conv2d(in_chans, embed_dim * self.spectral_size, kernel_size=patch_size, stride=patch_size, groups=self.spectral_size)
        else:
            raise ValueError(f"Unknown embedding_option: {embedding_option}")

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.embedding_option == 2 and self.spectral_size > 1:
            x_grouped = x.view(B, self.spectral_size, -1, H, W)
            output_groups = [self.proj(x_grouped[:, i]) for i in range(self.spectral_size)]
            x = torch.stack(output_groups, dim=1)  # (B, S, C_embed, H_p, W_p)
            x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, self.embed_dim)
        else:
            x = self.proj(x)
            B, _, H_p, W_p = x.shape
            x = x.view(B, self.spectral_size, self.embed_dim, H_p, W_p)
            x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, self.embed_dim)
        return self.norm(x)


class HSISwinTransformer(nn.Module):
    """Swin backbone for HSI with factorized spectral/spatial attention."""
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depths,
        num_heads,
        spatial_window_size,
        spectral_group_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        spatial_shift_size=None,
        pca_enabled=False,
        ablation_studies=None,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)

        # derive options
        _spatial_shift = spatial_shift_size if spatial_shift_size is not None else spatial_window_size // 2
        ablation = ablation_studies or {}
        embedding_option = ablation.get('embedding_option', 1)
        spectral_option = ablation.get('spectral_option', 1)

        if pca_enabled:
            embedding_option = 3
            self.spectral_size = in_chans
        else:
            self.spectral_size = (in_chans + spectral_group_size - 1) // spectral_group_size

        self.embed_dim = embed_dim
        self.patch_embed = AdaptiveHSIPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            spectral_group_size=spectral_group_size,
            norm_layer=norm_layer,
            embedding_option=embedding_option,
        )
        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        downsamples = []
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                downsamples.append(None)
            else:
                prev_dim = int(embed_dim * 2 ** (i_layer - 1))
                downsamples.append(HSIPatchMerging(dim=prev_dim, norm_layer=norm_layer, spectral_size=self.spectral_size))

        for i_layer in range(self.num_layers):
            stage_dim = int(embed_dim * 2 ** i_layer)
            self.layers.append(
                BasicLayer(
                    dim=stage_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=spatial_window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=downsamples[i_layer],
                    use_checkpoint=use_checkpoint,
                    use_factorized_attention=True,
                    spectral_size=self.spectral_size,
                    spatial_shift_size=_spatial_shift,
                    act_layer=nn.GELU,
                    spectral_option=spectral_option,
                )
            )

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, L, C_out), L = S_groups * H_tokens * W_tokens"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        H_p, W_p = self.patch_embed.patches_resolution
        for layer in self.layers:
            x, H_p, W_p = layer(x, H_p, W_p)

        x = self.norm(x)
        return x


class SpectralGlobalAttention(nn.Module):
    """Global attention along the spectral dimension."""
    def __init__(self, dim, num_heads, spectral_size, attn_drop, proj_drop, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, spectral_size, dim))
        trunc_normal_(self.spectral_pos_embed, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """x: (B*H*W, S, C)"""
        B_, S, C = x.shape
        x = x + self.spectral_pos_embed
        qkv = self.qkv(x).reshape(B_, S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------- NEW/REPLACED: Segformer (MIT) backbone wrapper ----------
class SegformerBackbone(nn.Module):
    """Wrapper for SegFormer / MIT backbones.

    Strategy:
      1) Try HuggingFace 'transformers' SegformerModel.from_pretrained(repo) to build backbone
         and return hidden_states reshaped to (B,C,H,W).
      2) Fallback to timm.create_model(...) + huggingface_hub snapshot download for weights
         (previous behavior).
    """
    def __init__(self, img_size=256, in_chans=3, model_name='mit_b0', pretrained=False, **kwargs):
        super().__init__()
        self.requested_pretrained = bool(pretrained)
        self.in_chans = in_chans
        self.repo_model_name = str(model_name)  # accept 'nvidia/mit-b0' or 'mit_b0'
        self.base_name = self.repo_model_name.split('/')[-1].replace('-', '_')

        # Try HuggingFace transformers SegformerModel first (preferred for HF repo artifacts)
        self.hf_model = None
        try:
            from transformers import SegformerModel, SegformerConfig
            # Try to load config from repo (if available) and ensure hidden states are returned
            try:
                cfg = SegformerConfig.from_pretrained(self.repo_model_name)
                cfg.output_hidden_states = True
            except Exception:
                # fallback to a minimal config object with output_hidden_states True
                cfg = SegformerConfig()
                cfg.output_hidden_states = True

            if self.requested_pretrained:
                try:
                    self.hf_model = SegformerModel.from_pretrained(self.repo_model_name, config=cfg, ignore_mismatched_sizes=True)
                except Exception:
                    # fallback: create model from config (no pretrained weights)
                    self.hf_model = SegformerModel(cfg)
            else:
                self.hf_model = SegformerModel(cfg)

            # if hf_model was created, set num/out channels from its encoder hidden sizes
            if self.hf_model is not None:
                # The HF Segformer encoder exposes embed_dims / hidden_sizes via config
                cfg = getattr(self.hf_model, "config", None)
                if cfg is not None and hasattr(cfg, "embed_dims"):
                    self.out_channels = list(cfg.embed_dims)
                else:
                    # best-effort fallback
                    self.out_channels = getattr(self.hf_model.config, "hidden_sizes", None) or getattr(self.hf_model.config, "embed_dims", None) or [256, 512, 1024]
                if isinstance(self.out_channels, int):
                    self.out_channels = [self.out_channels]
                self.num_features = int(self.out_channels[-1])
                return  # constructed via transformers -> done
        except Exception:
            # transformers not available or failed -> fall back to timm path below
            self.hf_model = None

        # FALLBACK: use timm architecture and optionally load HF weights into it
        if timm is None:
            raise ImportError(
                "Neither HuggingFace 'transformers' nor 'timm' are available to build SegFormer backbone. "
                "Install one of them: pip install transformers  OR  pip install timm"
            )

        # Try to create timm model (architecture). If timm doesn't know the model name, create without pretrained.
        try:
            self.timm_model = timm.create_model(self.base_name, pretrained=self.requested_pretrained, features_only=True, in_chans=self.in_chans)
        except Exception as e_timm_create:
            try:
                self.timm_model = timm.create_model(self.base_name, pretrained=False, features_only=True, in_chans=self.in_chans)
            except Exception as e_create_no_pre:
                raise RuntimeError(f"Failed to create timm model '{self.base_name}': {e_timm_create} / {e_create_no_pre}")

            # If pretrained requested, try HF hub to fetch weights
            if self.requested_pretrained:
                try:
                    from huggingface_hub import snapshot_download
                except Exception as e_hf_import:
                    raise ImportError(
                        "Requested pretrained weights but 'huggingface_hub' not installed. Install via: pip install huggingface_hub"
                    ) from e_hf_import

                try:
                    repo_dir = snapshot_download(repo_id=self.repo_model_name)
                except Exception as e_snap:
                    raise RuntimeError(f"Failed to download HF repo '{self.repo_model_name}': {e_snap}") from e_snap

                # search for common weight filenames
                candidates = []
                preferred_names = ['pytorch_model.bin', 'model.pth', 'model.pt', f"{self.base_name}.pth", f"{self.base_name}.pt"]
                for fname in preferred_names:
                    p = os.path.join(repo_dir, fname)
                    if os.path.exists(p):
                        candidates.append(p)

                if not candidates:
                    candidates = glob.glob(os.path.join(repo_dir, '**', '*.bin'), recursive=True) + \
                                 glob.glob(os.path.join(repo_dir, '**', '*.pth'), recursive=True) + \
                                 glob.glob(os.path.join(repo_dir, '**', '*.pt'), recursive=True)
                    candidates = sorted(set(candidates))

                if not candidates:
                    raise RuntimeError(f"No weight files (*.bin, *.pth, *.pt) found in HF repo snapshot: {repo_dir}")

                weight_path = candidates[0]
                try:
                    sd = torch.load(weight_path, map_location='cpu')
                except Exception as e_load:
                    raise RuntimeError(f"Failed to load weights file '{weight_path}': {e_load}") from e_load

                if isinstance(sd, dict) and ('model' in sd or 'state_dict' in sd):
                    if 'model' in sd and isinstance(sd['model'], dict):
                        sd = sd['model']
                    elif 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                        sd = sd['state_dict']

                try:
                    self.timm_model.load_state_dict(sd, strict=False)
                except Exception as e_load_sd:
                    try:
                        cleaned = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
                        self.timm_model.load_state_dict(cleaned, strict=False)
                    except Exception:
                        raise RuntimeError(f"Failed to load HF weights into timm model (path={weight_path}): {e_load_sd}")

        # determine out channels from timm feature_info
        self.out_channels = None
        try:
            if hasattr(self.timm_model, 'feature_info'):
                info = self.timm_model.feature_info
                try:
                    self.out_channels = list(info.channels())
                except Exception:
                    self.out_channels = [f['num_chs'] for f in getattr(info, 'info', [])] if hasattr(info, 'info') else None
        except Exception:
            self.out_channels = None

        if self.out_channels is None:
            nf = getattr(self.timm_model, 'num_features', None)
            if isinstance(nf, (list, tuple)):
                self.out_channels = list(nf)
            elif isinstance(nf, int):
                self.out_channels = [nf]
            else:
                self.out_channels = [256]

        self.num_features = int(self.out_channels[-1])

    def forward_features(self, x):
        """Return list of feature maps (B,C,H,W).

        If hf_model is present (transformers), call it and reshape hidden_states -> feature maps.
        Otherwise call timm feature extractor (features_only).
        """
        if self.hf_model is not None:
            # transformers SegformerModel expects 'pixel_values' or raw tensor as first arg
            # Ensure output_hidden_states True either via config or runtime call
            outputs = self.hf_model(x, output_hidden_states=True)
            hidden = getattr(outputs, "hidden_states", None)
            if hidden is None:
                # try last_hidden_state as single-level fallback
                last = getattr(outputs, "last_hidden_state", None)
                if last is None:
                    raise RuntimeError("Transformers SegformerModel returned no hidden states.")
                hidden = (last,)

            feats = []
            # skip initial embedding (if present) and use deeper feature maps; include all to be safe
            for hs in hidden:
                if hs is None:
                    continue
                # hs: (B, L, C)
                # normalize hs into (B, L, C) where L = sequence length (H*W or S*H*W)
                if not isinstance(hs, torch.Tensor):
                    raise TypeError("Expected tensor for 'hs' features")
                if hs.ndim == 3:
                    B, L, C = hs.shape
                elif hs.ndim == 4:
                    # assume (B, C, H, W) -> convert to (B, H*W, C)
                    B, C, H, W = hs.shape
                    L = H * W
                    hs = hs.permute(0, 2, 3, 1).reshape(B, L, C)
                elif hs.ndim == 5:
                    # e.g. (B, S, H, W, C) -> flatten spatial+S into sequence
                    B, S, H, W, C = hs.shape
                    L = S * H * W
                    hs = hs.view(B, L, C)
                else:
                    raise ValueError(f"Unexpected hs.ndim={hs.ndim}; expected 3/4/5 dims")

                feat = hs.transpose(1, 2).reshape(B, C, H, W)
                feats.append(feat)
            if not feats:
                raise RuntimeError("No valid hidden-states could be converted to feature maps from HuggingFace SegformerModel.")
            return feats

        # timm feature extractor path
        if hasattr(self, 'timm_model'):
            feats = self.timm_model.forward_features(x) if hasattr(self.timm_model, "forward_features") else self.timm_model(x)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]
            return feats

        raise RuntimeError("SegformerBackbone has no valid internal model (transformers or timm).")
#
def build_swin_transformer(config=None):
    """Factory to build Swin (RGB/HSI) or SegFormer (RGB) backbones."""
    modality = config['modality']
    backbone_config = config.get('backbone', {}).copy()
    pca_enabled = config.get('augmentation', {}).get('pca_transform', {}).get('enable', False)

    # NEW: route SegFormer requests
    btype = str(backbone_config.get('type', '')).lower()
    model_name = str(backbone_config.get('model_name', ''))
    looks_segformer = (
        btype == 'segformer'
        or 'segformer' in model_name.lower()
        or 'mit' in model_name.lower()
        or model_name.lower().startswith('nvidia/')
    )
    if modality == 'rgb' and looks_segformer:
        return SegformerBackbone(
            img_size=backbone_config.get('img_size', 256),
            in_chans=backbone_config.get('in_chans', 3),
            model_name=model_name or 'nvidia/mit-b0',
            pretrained=bool(backbone_config.get('pretrained', False)),
        )

    hsi_mode = backbone_config.get('hsi_mode', 'factorized')

    if modality == 'hsi' and hsi_mode == 'rgb_adapter':
        return HSIRGBAdapterSwin(
            original_in_chans=backbone_config.get('in_chans', 225),
            img_size=backbone_config.get('img_size', 256),
            patch_size=backbone_config.get('patch_size', 4),
            embed_dim=backbone_config.get('embed_dim', 96),
            depths=backbone_config.get('depths', [2, 2, 6, 2]),
            num_heads=backbone_config.get('num_heads', [3, 6, 12, 24]),
            window_size=backbone_config.get('spatial_window_size', backbone_config.get('window_size', 7)),
            mlp_ratio=backbone_config.get('mlp_ratio', 4.0),
            drop_rate=backbone_config.get('drop_rate', 0.0),
            attn_drop_rate=backbone_config.get('attn_drop_rate', 0.0),
            drop_path_rate=backbone_config.get('drop_path_rate', 0.1),
            qkv_bias=backbone_config.get('qkv_bias', True),
            adapter_kernel_size=backbone_config.get('adapter_kernel_size', 1),
        )

    if modality == 'rgb':
        return SwinTransformer(**backbone_config)

    if modality == 'hsi':
        backbone_config['pca_enabled'] = pca_enabled
        backbone_config['ablation_studies'] = config.get('ablation_studies', {})
        return HSISwinTransformer(**backbone_config)

    raise ValueError(f"Unsupported modality: {modality}")

class HSIRGBAdapterSwin(SwinTransformer):
    def __init__(
        self,
        original_in_chans,
        img_size,
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        adapter_kernel_size=1,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer
        )
        pad = adapter_kernel_size // 2
        self.adapter = nn.Sequential(
            nn.Conv2d(original_in_chans, 3, adapter_kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.hsi_mode = 'rgb_adapter'


