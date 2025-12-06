# Inspired by UNet and SENet

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) UNet-style decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, dropout_rate=0.0):
        super().__init__()
        self.has_skip = skip_channels > 0

        # upsample
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        # convs
        conv_layers = []
        if self.has_skip:
            conv_layers.extend([
                nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        else:
            # no skip: input is already 'out_channels' from upsample
            conv_layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        if dropout_rate > 0:
            conv_layers.append(nn.Dropout2d(p=dropout_rate))

        conv_layers.extend([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        self.conv_blocks = nn.Sequential(*conv_layers)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.has_skip and skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv_blocks(x)


# 2) RGB/vanilla 2D head
class UNetHead(nn.Module):
    def __init__(self, in_dims, config, use_checkpoint=False):
        super().__init__()
        in_dims_reversed = in_dims[::-1]
        self.num_classes = config['num_classes']
        decoder_channels = config['decoder_channels']
        embed_dim = config.get('embed_dim', decoder_channels[0])
        self.use_checkpoint = use_checkpoint
        self.dropout_rate = config.get('dropout', 0.0)

        self.adapters = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for dim in in_dims_reversed])

        self.decoder_blocks = nn.ModuleList()
        prev_out_channels = embed_dim
        for i, current_out in enumerate(decoder_channels):
            has_skip = (i + 1) < len(in_dims_reversed)
            skip_ch = embed_dim if has_skip else 0
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=prev_out_channels,
                    out_channels=current_out,
                    skip_channels=skip_ch,
                    dropout_rate=self.dropout_rate,
                )
            )
            prev_out_channels = current_out

        self.classifier = nn.Conv2d(decoder_channels[-1], self.num_classes, 1)

    def forward(self, features):
        # features: [shallowest, ..., deepest] or a single tensor
        if isinstance(features, torch.Tensor):
            features = [features]
        feats = [adapt(f) for adapt, f in zip(self.adapters, features[::-1])]
        x = feats[0]
        for i, blk in enumerate(self.decoder_blocks):
            skip = feats[i + 1] if (i + 1) < len(feats) else None
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, skip)
            else:
                x = blk(x, skip)
        return self.classifier(x)


# 3) Spectral reducers for HSI head
class SpectralSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, S, H, W]
        _, C, S, H, W = x.shape
        w = self.fc(x.mean(dim=(3, 4)))                    # [B, C, S]
        x = (x * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2) # -> [B, C, H, W]
        return x


class LearnableWeightedReducer(nn.Module):
    """Learnable weighted sum along spectral dimension."""
    def __init__(self, channels, expected_s):
        super().__init__()
        self.spectral_weights = nn.Parameter(torch.ones(1, 1, expected_s, 1, 1))

    def forward(self, x):
        # x: [B, C, S, H, W]
        weights = F.softmax(self.spectral_weights, dim=2)
        return (x * weights).sum(dim=2)


# 4) HSI hybrid head (spectral reduction + 2D UNet blocks)
class HSIUNetHead(nn.Module):
    def __init__(self, in_dims, config, use_checkpoint=False, ablation_studies=None):
        super().__init__()
        in_dims_reversed = in_dims[::-1]
        self.num_classes = config['num_classes']
        decoder_channels = config['decoder_channels']
        embed_dim = config.get('embed_dim', decoder_channels[0])
        self.use_checkpoint = use_checkpoint
        self.dropout_rate = config.get('dropout', 0.0)

        se_option = (ablation_studies or {}).get('SE_option', 1)
        expected_s = config.get('expected_s', None)

        self.spectral_reducers = nn.ModuleList()
        if se_option == 1:
            self.spectral_reducers = nn.ModuleList([SpectralSE(dim) for dim in in_dims_reversed])
        elif se_option == 2:
            if expected_s is None:
                raise ValueError("LearnableWeightedReducer requires 'expected_s' in config.")
            self.spectral_reducers = nn.ModuleList([LearnableWeightedReducer(dim, expected_s) for dim in in_dims_reversed])
        else:
            raise ValueError(f"Unknown SE_option: {se_option}")

        # 1x1 adapters after spectral reduction
        self.adapters = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for dim in in_dims_reversed])

        self.decoder_blocks = nn.ModuleList()
        prev_out_channels = embed_dim
        for i, current_out in enumerate(decoder_channels):
            has_skip = (i + 1) < len(in_dims_reversed)
            skip_ch = embed_dim if has_skip else 0
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=prev_out_channels,
                    out_channels=current_out,
                    skip_channels=skip_ch,
                    dropout_rate=self.dropout_rate,
                )
            )
            prev_out_channels = current_out

        self.classifier = nn.Conv2d(decoder_channels[-1], self.num_classes, 1)

    def forward(self, features):
        # features: list of 5D tensors [shallowest, ..., deepest], each (B, C_s, S, H, W)
        if isinstance(features, torch.Tensor):
            features = [features]

        feats = []
        for i, f in enumerate(features[::-1]):
            f_reduced = self.spectral_reducers[i](f)  # -> (B, C_s, H, W)
            feats.append(self.adapters[i](f_reduced)) # -> (B, embed_dim, H, W)

        x = feats[0]
        for i, blk in enumerate(self.decoder_blocks):
            skip = feats[i + 1] if (i + 1) < len(feats) else None
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, skip)
            else:
                x = blk(x, skip)
        return self.classifier(x)


# 5) Head factory
def build_segmentation_head(config, feature_dim):
    # Prefer explicit encoder channels (from config head) -> backbone out_channels (SegFormer/timm) ->
    # fallback to embed_dim multiples.
    head_cfg = config.get('head', {}) if isinstance(config.get('head', {}), dict) else {}
    encoder_channels = head_cfg.get('encoder_channels') or config.get('backbone', {}).get('out_channels')
    if encoder_channels:
        # ensure list and keep order [stage0, stage1, stage2, stage3]
        in_dims = list(encoder_channels)
    else:
        base_dim = config.get('backbone', {}).get('embed_dim', feature_dim or 96)
        in_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

    head_cfg = config['head'].copy()
    head_cfg['num_classes'] = config['num_classes']
    ablation_studies = config.get('ablation_studies', {})

    # Accept encoder_channels if decoder_channels not present
    if 'decoder_channels' not in head_cfg:
        enc = head_cfg.get('encoder_channels', None)
        if enc and isinstance(enc, (list, tuple)) and len(enc) >= 3:
            # simple rule: use last three reversed or shrink
            head_cfg['decoder_channels'] = [enc[-1] // 2, enc[-2] // 2, enc[-3] // 2]
        else:
            head_cfg['decoder_channels'] = [256, 128, 64]

    expected_s = None
    if config.get('modality') == 'hsi':
        in_chans = config.get('backbone', {}).get('in_chans', None)
        group_sz = config.get('backbone', {}).get('spectral_group_size', None)
        if in_chans and group_sz and group_sz > 0:
            expected_s = (in_chans + group_sz - 1) // group_sz
    head_cfg['expected_s'] = expected_s

    if config['modality'] == 'hsi':
        return HSIUNetHead(in_dims, head_cfg, ablation_studies=ablation_studies)
    return UNetHead(in_dims, head_cfg)


class FusedFeatureFusionHead(nn.Module):
    """
    Decoder for fused cross-attention features.
    Expects fused_features with keys: 'stage_0'...'stage_3'.
    """
    def __init__(self, head_config, num_classes, fusion_stages):
        super().__init__()
        self.head_config = head_config
        self.num_classes = num_classes
        self.fusion_stages = fusion_stages
        self.dropout_rate = head_config.get('dropout', 0.0)

        decoder_channels = head_config.get('decoder_channels', [256, 128, 64])
        decoder_first_dim = decoder_channels[0]

        self.decoder_blocks = nn.ModuleList([
            # decoder0: stage_3 + stage_2
            DecoderBlock(decoder_first_dim, decoder_channels[0], skip_channels=decoder_first_dim, dropout_rate=self.dropout_rate),
            # decoder1: prev + stage_1
            DecoderBlock(decoder_channels[0], decoder_channels[1], skip_channels=decoder_first_dim, dropout_rate=self.dropout_rate),
            # decoder2: prev + stage_0
            DecoderBlock(decoder_channels[1], decoder_channels[2], skip_channels=decoder_first_dim, dropout_rate=self.dropout_rate),
        ])
        self.classifier = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)

    def forward(self, fused_features):
        required = ['stage_0', 'stage_1', 'stage_2', 'stage_3']
        missing = [k for k in required if k not in fused_features]
        if missing:
            raise ValueError(f"Missing required stages in fused_features: {missing}")

        x = fused_features['stage_3']
        x = self.decoder_blocks[0](x, fused_features['stage_2'])
        x = self.decoder_blocks[1](x, fused_features['stage_1'])
        x = self.decoder_blocks[2](x, fused_features['stage_0'])
        return self.classifier(x)