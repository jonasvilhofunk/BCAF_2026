import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_head import SpectralSE


class BidirectionalCrossAttention(nn.Module):
    """
    Cross-scale spectral fusion with NxN parent–child mapping (N inferred at runtime).
    Assumes RGB is the fine grid (H_rgb, W_rgb), HSI is the coarse grid (H_hsi, W_hsi),
    and H_rgb = N*H_hsi, W_rgb = N*W_hsi with N >= 2.

    Will assert-fail if N == 1 (same resolution) or if ratios are inconsistent.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        num_cross_attention_layers=1,
        mlp_ratio=4.0,
        dropout=0.0,
        fusion_direction='bidirectional'
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_cross_attention_layers
        self.num_heads = num_heads

        self.cross_attention_layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                fusion_direction=fusion_direction,  # NEW
            ) for _ in range(self.num_layers)
        ])

    def forward(self, rgb_feat, hsi_feat):
        """
        Args:
            rgb_feat: (B, C, H_rgb, W_rgb)  # fine grid
            hsi_feat: (B, C, S, H_hsi, W_hsi)  # coarse grid

        Returns:
            z_fused: (B, C, H_rgb, W_rgb)
        """
        assert rgb_feat.ndim == 4, f"Expected RGB features to be 4D, got {rgb_feat.ndim}D"
        assert hsi_feat.ndim == 5, f"Expected HSI features to be 5D, got {hsi_feat.ndim}D"

        # Do NOT upsample HSI beforehand; pass native grids.
        z_rgb = rgb_feat.permute(0, 2, 3, 1)           # (B, H_rgb, W_rgb, C)
        z_hsi = hsi_feat.permute(0, 3, 4, 2, 1)        # (B, H_hsi, W_hsi, S, C)

        z_fused = None
        for layer in self.cross_attention_layers:
            z_fused = layer(z_rgb, z_hsi)              # (B, C, H_rgb, W_rgb)

        return z_fused


class BidirectionalCrossAttentionBlock(nn.Module):
    """
    Parallel cross-attention with NxN parent–child mapping (N inferred):


      - RGB→HSI: the N^2 RGB children query S HSI spectral tokens of their parent
      - HSI→RGB: S HSI tokens query their N^2 RGB children

      - HSI branch is collapsed via SpectralSE and upsampled (nearest) to the fine grid for fusion

    """
    def __init__(self, d_model, num_heads, mlp_ratio=4., dropout=0., 
                 fusion_direction='bidirectional'):  # NEW parameter
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.fusion_direction = fusion_direction  # 'bidirectional', 'rgb_to_hsi', 'hsi_to_rgb'

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention projections
        if fusion_direction in ['bidirectional', 'rgb_to_hsi']:
            self.rgb_to_hsi_q = nn.Linear(d_model, d_model)
            self.rgb_to_hsi_k = nn.Linear(d_model, d_model)
            self.rgb_to_hsi_v = nn.Linear(d_model, d_model)
            self.rgb_to_hsi_proj = nn.Linear(d_model, d_model)
        
        if fusion_direction in ['bidirectional', 'hsi_to_rgb']:
            self.hsi_to_rgb_q = nn.Linear(d_model, d_model)
            self.hsi_to_rgb_k = nn.Linear(d_model, d_model)
            self.hsi_to_rgb_v = nn.Linear(d_model, d_model)
            self.hsi_to_rgb_proj = nn.Linear(d_model, d_model)

        # Layer norms
        self.norm_rgb_1 = nn.LayerNorm(d_model)
        self.norm_rgb_2 = nn.LayerNorm(d_model)
        self.norm_hsi_1 = nn.LayerNorm(d_model)
        self.norm_hsi_2 = nn.LayerNorm(d_model)

        # FFNs
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.rgb_ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.hsi_ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

        # SpectralSE for final spectral fusion
        self.spectral_se = SpectralSE(channels=d_model, reduction=8)

        # Fusion norms and gate
        self.norm_rgb_fuse = nn.LayerNorm(d_model)
        self.norm_hsi_fuse = nn.LayerNorm(d_model)
        self.fuse_ln = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.zeros(d_model))  # per-channel gate for HSI

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _infer_ratio(H_rgb, W_rgb, H_hsi, W_hsi):
        assert H_rgb % H_hsi == 0 and W_rgb % W_hsi == 0, \
            f"RGB/HSI sizes not integer multiples: RGB {(H_rgb, W_rgb)} vs HSI {(H_hsi, W_hsi)}"
        r_h, r_w = H_rgb // H_hsi, W_rgb // W_hsi
        assert r_h == r_w, f"Mismatched ratios: r_h={r_h}, r_w={r_w}"
        r = r_h
        # Allow r == 1 (same resolution) now
        if r == 1:
            # Optional: print once (remove if noisy)
            # print("Info: RGB and HSI share the same spatial resolution (r=1). Using 1:1 cross-attention.")
            pass
        return r

    def per_pixel_cross_attention(self, q, k, v):
        """
        Args:
            q: (B, H, W, seq_len_q, C)
            k: (B, H, W, seq_len_k, C)
            v: (B, H, W, seq_len_k, C)
        Returns:
            out: (B, H, W, seq_len_q, C)
        """
        B, H, W, Lq, C = q.shape
        Lk = k.shape[3]
        q = q.reshape(B * H * W, Lq, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B * H * W, Lk, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B * H * W, Lk, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, H, W, Lq, C)
        return out

    def _extract_children(self, z_rgb_norm1, H_hsi, W_hsi, r):
        """
        Returns r^2 children per HSI parent using pixel_unshuffle when r>1.
        For r == 1 (same resolution), returns a singleton child per pixel:
            (B, H_hsi, W_hsi, 1, C)
        """
        B, H_rgb, W_rgb, C = z_rgb_norm1.shape
        assert H_rgb == H_hsi * r and W_rgb == W_hsi * r, "Ratio mismatch in _extract_children"
        if r == 1:
            # Each pixel is its own 'child' ⇒ add a length-1 sequence dimension
            return z_rgb_norm1.view(B, H_hsi, W_hsi, 1, C)
        x = z_rgb_norm1.permute(0, 3, 1, 2)                 # (B, C, H_rgb, W_rgb)
        y = F.pixel_unshuffle(x, downscale_factor=r)        # (B, C*r*r, H_hsi, W_hsi)
        y = y.view(B, C, r * r, H_hsi, W_hsi).permute(0, 3, 4, 2, 1).contiguous()  # (B, H_hsi, W_hsi, r*r, C)
        return y

    def _children_to_fine_grid(self, x_patch, r):
        """
        x_patch: (B, H_hsi, W_hsi, r*r, C) -> (B, H_rgb, W_rgb, C)
        r == 1: identity reshape.
        """
        if r == 1:
            # Remove the length-1 children axis
            return x_patch[..., 0, :]  # (B, H, W, C)
        B, Hc, Wc, rr, C = x_patch.shape
        assert rr == r * r, "Unexpected children count"
        x = x_patch.permute(0, 4, 3, 1, 2).contiguous()     # (B, C, r*r, Hc, Wc)
        x = x.view(B, C * r * r, Hc, Wc)                    # (B, C*r*r, Hc, Wc)
        x = F.pixel_shuffle(x, upscale_factor=r)            # (B, C, Hc*r, Wc*r)
        x = x.permute(0, 2, 3, 1).contiguous()              # (B, H_rgb, W_rgb, C)
        return x

    def forward(self, z_rgb, z_hsi):
        """
        Args:
            z_rgb: (B, H_rgb, W_rgb, C) - fine (or same) grid
            z_hsi: (B, H_hsi, W_hsi, S, C) - coarse (or same) grid
        Returns:
            z_fused: (B, C, H_rgb, W_rgb)
        """
        B, H_rgb, W_rgb, C = z_rgb.shape
        _, H_hsi, W_hsi, S, _ = z_hsi.shape

        r = self._infer_ratio(H_rgb, W_rgb, H_hsi, W_hsi)

        z_rgb_norm1 = self.norm_rgb_1(z_rgb)
        z_hsi_norm1 = self.norm_hsi_1(z_hsi)

        rgb_children = self._extract_children(z_rgb_norm1, H_hsi, W_hsi, r)  # (B, H_hsi, W_hsi, r*r (or 1), C)

        # RGB enhancement (RGB queries HSI)
        if self.fusion_direction in ['bidirectional', 'rgb_to_hsi']:
            q_rgb = self.rgb_to_hsi_q(rgb_children)
            k_hsi = self.rgb_to_hsi_k(z_hsi_norm1)
            v_hsi = self.rgb_to_hsi_v(z_hsi_norm1)
            rgb_attn_out_patch = self.per_pixel_cross_attention(q_rgb, k_hsi, v_hsi)
            rgb_attn_out = self._children_to_fine_grid(rgb_attn_out_patch, r)
            rgb_attn_out = self.rgb_to_hsi_proj(rgb_attn_out)
            z_rgb_enhanced = z_rgb + rgb_attn_out
        else:
            z_rgb_enhanced = z_rgb  # Skip this direction
        
        # HSI enhancement (HSI queries RGB)
        if self.fusion_direction in ['bidirectional', 'hsi_to_rgb']:
            q_hsi = self.hsi_to_rgb_q(z_hsi_norm1)
            k_rgb = self.hsi_to_rgb_k(rgb_children)
            v_rgb = self.hsi_to_rgb_v(rgb_children)
            hsi_attn_out = self.per_pixel_cross_attention(q_hsi, k_rgb, v_rgb)
            hsi_attn_out = self.hsi_to_rgb_proj(hsi_attn_out)
            z_hsi_enhanced = z_hsi + hsi_attn_out
        else:
            z_hsi_enhanced = z_hsi  # Skip this direction
        
        z_rgb_final = z_rgb_enhanced + self.rgb_ffn(self.norm_rgb_2(z_rgb_enhanced))
        z_hsi_final = z_hsi_enhanced + self.hsi_ffn(self.norm_hsi_2(z_hsi_enhanced))

        z_hsi_5d = z_hsi_final.permute(0, 4, 3, 1, 2)               # (B, C, S, H_hsi, W_hsi)
        z_hsi_collapsed = self.spectral_se(z_hsi_5d)                # (B, C, H_hsi, W_hsi)
        z_hsi_collapsed = z_hsi_collapsed.permute(0, 2, 3, 1)       # (B, H_hsi, W_hsi, C)

        z_rgb_norm_fuse = self.norm_rgb_fuse(z_rgb_final)
        z_hsi_norm_fuse = self.norm_hsi_fuse(z_hsi_collapsed)

        # Upsample only if r > 1; else identity
        if r > 1:
            z_hsi_up = z_hsi_norm_fuse.permute(0, 3, 1, 2)
            z_hsi_up = F.interpolate(z_hsi_up, scale_factor=r, mode='nearest')
            z_hsi_up = z_hsi_up.permute(0, 2, 3, 1)
        else:
            z_hsi_up = z_hsi_norm_fuse  # same spatial size

        gate = torch.sigmoid(self.alpha).view(1, 1, 1, -1)
        z_fused = self.fuse_ln(z_rgb_norm_fuse + gate * z_hsi_up)
        return z_fused.permute(0, 3, 1, 2)