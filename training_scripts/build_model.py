from collections import OrderedDict
from pathlib import Path
import warnings
import yaml
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Prefer local modules; fall back to old package paths if needed
from .backbones import build_swin_transformer
from .segmentation_head import build_segmentation_head, FusedFeatureFusionHead
from .cross_attention_fusion import BidirectionalCrossAttention




class UnimodalSegmentationModel(nn.Module):
    """Unimodal segmentation: Swin backbone + segmentation head."""
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        if num_classes is not None:
            self.config['num_classes'] = num_classes

        # Build backbone via factory (may return SwinTransformer, HSISwinTransformer, or SegformerBackbone)
        self.backbone = build_swin_transformer(config)
        # expose num_features for head construction (fallbacks if not present)
        self.num_features = getattr(self.backbone, 'num_features', None)
        if self.num_features is None:
            # attempt common fallbacks
            if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone, 'embed_dim'):
                self.num_features = int(self.backbone.embed_dim * 2 ** (getattr(self.backbone, 'num_layers', 4) - 1))
            else:
                raise RuntimeError("Backbone did not expose 'num_features' attribute; cannot construct head.")
        # If the backbone exposes per-stage output channels (SegFormer / timm features_only),
        # make them available to the head factory so the adapters are constructed with the
        # correct input channels.
        if hasattr(self.backbone, 'out_channels'):
            try:
                config.setdefault('backbone', {})['out_channels'] = list(self.backbone.out_channels)
            except Exception:
                # fallback if out_channels is an int or other type
                config.setdefault('backbone', {})['out_channels'] = getattr(self.backbone, 'out_channels')

        # --- NEW: choose RGB-style head when backbone is SegFormer-like but modality is HSI ---
        head_cfg = config
        # SegFormer HSI -> RGB head (existing)
        if self.config.get('modality') == 'hsi' and not hasattr(self.backbone, 'patch_embed'):
            head_cfg = config.copy()
            head_cfg['modality'] = 'rgb'
            print("Info: building RGB-style head (SegFormer HSI assumed PCA->3ch).")
        # NEW: rgb_adapter HSI -> treat as RGB head
        elif self.config.get('modality') == 'hsi' and getattr(self.backbone, 'hsi_mode', '') == 'rgb_adapter':
            head_cfg = config.copy()
            head_cfg['modality'] = 'rgb'
            print("Info: building RGB-style head for HSI rgb_adapter path.")

        # build segmentation head using channels from backbone
        self.seg_head = build_segmentation_head(head_cfg, self.num_features)

        # Gate legacy aliases behind a flag (default False)
        register_legacy_aliases = config.get('backbone', {}).get('register_legacy_aliases', False)

        # REMOVE legacy alias registration by default
        if register_legacy_aliases and hasattr(self.backbone, 'patch_embed'):
            self.patch_embed = self.backbone.patch_embed
        if register_legacy_aliases and hasattr(self.backbone, 'pos_drop'):
            self.pos_drop = self.backbone.pos_drop
        if register_legacy_aliases and hasattr(self.backbone, 'layers'):
            self.layers = self.backbone.layers
        if register_legacy_aliases and hasattr(self.backbone, 'norm'):
            self.norm = self.backbone.norm

    def forward_features(self, x):
        # Early route for rgb_adapter (apply adapter then treat as RGB Swin)
        if getattr(self.backbone, 'hsi_mode', '') == 'rgb_adapter':
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError("HSI input must be 4D Tensor [B,C,H,W].")
            x = self.backbone.adapter(x)  # (B,3,H,W)
            # Standard RGB Swin feature extraction
            pe = self.backbone.patch_embed
            layers = self.backbone.layers
            norm = self.backbone.norm
            pos_drop = self.backbone.pos_drop
            feats = []
            x = pe(x)
            H, W = pe.patches_resolution
            if pos_drop is not None:
                x = pos_drop(x)
            for i, layer in enumerate(layers):
                x, H, W = layer(x, H, W)
                C = int(pe.embed_dim * (2 ** i))
                feats.append(x.transpose(1, 2).reshape(-1, C, H, W))
            x = norm(x)
            C_final = int(pe.embed_dim * (2 ** (len(layers) - 1)))
            feats[-1] = x.transpose(1, 2).reshape(-1, C_final, H, W)
            return feats

        # SegFormer-like path (no patch_embed)
        if not hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone, 'forward_features'):
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError("Input must be 4D Tensor [B,C,H,W].")
            B, C, H, W = x.shape
            expected_in_chans = int(self.config.get('backbone', {}).get('in_chans', 3))
            if C != expected_in_chans:
                if C > expected_in_chans:
                    x = x[:, :expected_in_chans, :, :].contiguous()
                    warnings.warn(f"Input had {C} channels; truncating to {expected_in_chans} for SegFormer-like backbone.")
                else:
                    pad = torch.zeros((B, expected_in_chans - C, H, W), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=1).contiguous()
                    warnings.warn(f"Input had {C} channels; padding to {expected_in_chans} for SegFormer-like backbone.")
            return self.backbone.forward_features(x)

        # Swin-style HSI path (backbone exposes patch_embed/layers/norm)
        if self.config.get('modality', 'rgb') == 'hsi':
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError("HSI input must be 4D Tensor [B,C,H,W].")

            pe = getattr(self.backbone, 'patch_embed', None)
            layers = getattr(self.backbone, 'layers', None)
            norm = getattr(self.backbone, 'norm', None)
            pos_drop = getattr(self.backbone, 'pos_drop', None)
            if pe is None or layers is None or norm is None:
                raise AttributeError("HSI path expects backbone to expose patch_embed/layers/norm.")

            B, C, H_in, W_in = x.shape
            expected_in_chans = int(self.config.get('backbone', {}).get('in_chans', getattr(pe, 'in_chans', C)))

            if C == expected_in_chans + 3:
                x = x[:, 3:, :, :].contiguous()
                C = x.shape[1]

            if C != expected_in_chans:
                if C > expected_in_chans:
                    x = x[:, -expected_in_chans:, :, :].contiguous()
                else:
                    pad = torch.zeros((B, expected_in_chans - C, H_in, W_in), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=1).contiguous()
                warnings.warn(f"Adjusted HSI channels from {C} to expected {expected_in_chans}.")

            x = pe(x)
            if pos_drop is not None:
                x = pos_drop(x)

            S = getattr(pe, 'spectral_size', 1)
            H, W = pe.patches_resolution
            feats = []
            for i, layer in enumerate(layers):
                x, H, W = layer(x, H, W)
                current_C = int(pe.embed_dim * (2 ** i))
                x_out = x.view(-1, S, H, W, current_C).permute(0, 4, 1, 2, 3).contiguous()
                feats.append(x_out)

            x = norm(x)
            final_C = int(pe.embed_dim * (2 ** (len(layers) - 1)))
            x_final = x.view(-1, S, H, W, final_C).permute(0, 4, 1, 2, 3).contiguous()
            if feats:
                feats[-1] = x_final
            else:
                feats.append(x_final)
            return feats

        # Swin-style RGB path
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("RGB input must be 4D Tensor [B,C,H,W].")

        feats = []
        x = self.backbone.patch_embed(x)
        Wh, Ww = self.backbone.patch_embed.patches_resolution
        if self.backbone.pos_drop is not None:
            x = self.backbone.pos_drop(x)
        for layer in self.backbone.layers:
            out = layer(x, Wh, Ww)
            x, Wh, Ww = out if isinstance(out, tuple) else (out, Wh, Ww)
            B, L, C = x.shape
            feats.append(x.transpose(1, 2).reshape(B, C, Wh, Ww))

        x = self.backbone.norm(x)
        B, L, C = x.shape
        final_x = x.transpose(1, 2).reshape(B, C, Wh, Ww)
        if feats:
            feats[-1] = final_x
        else:
            feats.append(final_x)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.seg_head(feats)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

    def backbone_forward(self, x):
        """Expose backbone feature extraction (list of per-stage features)."""
        return self.forward_features(x)


def _load_unimodal_component(config_path, checkpoint_path, num_classes, device, name="component"):
    """Load an UnimodalSegmentationModel from config and optional checkpoint."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['num_classes'] = num_classes
    cfg['pretrained'] = {'use_pretrained': "None"}

    model = UnimodalSegmentationModel(cfg)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # strip DataParallel/DataLoader prefixes
        if isinstance(sd, dict):
            sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}

        has_backbone_prefix = any(k.startswith('backbone.') for k in sd.keys())
        looks_like_raw_backbone = any(
            key.startswith(prefix)
            for key in sd.keys()
            for prefix in ('patch_embed.', 'layers.', 'norm.', 'stages.', 'blocks.', 'pos_drop', 'pos_embed', 'cls_token')
        )

        missing = unexpected = []
        if looks_like_raw_backbone and not has_backbone_prefix:
            # Load straight into backbone
            missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        else:
            # Try whole-model load
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # If many missing and looks like raw backbone, try prefixing
            if len(missing) > max(10, int(len(sd) * 0.5)) and not has_backbone_prefix and looks_like_raw_backbone:
                sd_prefixed = {f'backbone.{k}': v for k, v in sd.items()}
                missing, unexpected = model.load_state_dict(sd_prefixed, strict=False)

        if missing or unexpected:
            print(f"Warning: partial load for {name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    elif checkpoint_path:
        print(f"Warning: checkpoint not found for {name} at {checkpoint_path}")

    model.to(device)
    return model


class LateLogitFusionModel(nn.Module):
    """Fuse logits from two unimodal models via 1x1 conv."""
    def __init__(self, fusion_config, num_classes, device):
        super().__init__()
        self.num_classes = num_classes

        self.rgb_model = _load_unimodal_component(
            fusion_config['rgb_model_config_path'],
            fusion_config['rgb_checkpoint_path'],
            num_classes, device, name="RGB"
        )
        self.hsi_model = _load_unimodal_component(
            fusion_config['hsi_model_config_path'],
            fusion_config['hsi_checkpoint_path'],
            num_classes, device, name="HSI"
        )
        self.fusion_layer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=True)

    def forward(self, inputs):
        rgb_logits = self.rgb_model(inputs['rgb'])
        hsi_logits = self.hsi_model(inputs['hsi'])
        if hsi_logits.shape[2:] != rgb_logits.shape[2:]:
            hsi_logits = F.interpolate(hsi_logits, size=rgb_logits.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion_layer(torch.cat((rgb_logits, hsi_logits), dim=1))


class CrossAttentionFusionModel(nn.Module):
    """Feature-level fusion with cross-attention at selected stages."""
    def __init__(self, main_config, num_classes, device):
        super().__init__()
        if BidirectionalCrossAttention is None:
            raise ImportError("BidirectionalCrossAttention not found. Ensure cross_attention_fusion module is available.")

        self.num_classes = num_classes
        self.rgb_model = _load_unimodal_component(
            main_config['rgb_model_config_path'], main_config['rgb_checkpoint_path'], num_classes, device, "RGB"
        )
        self.hsi_model = _load_unimodal_component(
            main_config['hsi_model_config_path'], main_config['hsi_checkpoint_path'], num_classes, device, "HSI"
        )

        fusion_cfg = main_config.get('fusion', {})
        self.fusion_stages = fusion_cfg.get('fusion_stages', [0, 1, 2, 3])
        scaling_factors = fusion_cfg.get('scaling_factors', [1.0, 1.0, 1.0, 1.0])
        num_ca_layers = fusion_cfg.get('num_cross_attention_layers', 1)
        fusion_direction = fusion_cfg.get('fusion_direction', 'bidirectional')  # NEW

        base_dim = self.rgb_model.config.get('embed_dim', 96)
        self.stage_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.all_stages = [0, 1, 2, 3]
        self.rgb_only_stages = [s for s in self.all_stages if s not in self.fusion_stages]

        # Heads per stage for Swin
        with open(main_config['hsi_model_config_path'], 'r') as f:
            hsi_cfg = yaml.safe_load(f)
        hsi_heads = hsi_cfg.get('backbone', {}).get('num_heads', [3, 6, 12, 24])

        # Create cross-attention modules with direction
        self.cross_attention_modules = nn.ModuleDict()
        for s in self.fusion_stages:
            heads = hsi_heads[s] if s < len(hsi_heads) else hsi_heads[-1]
            self.cross_attention_modules[f'stage_{s}'] = BidirectionalCrossAttention(
                d_model=self.stage_dims[s], 
                num_heads=heads, 
                num_cross_attention_layers=num_ca_layers,
                fusion_direction=fusion_direction  # NEW
            )

        # Optional pre-CA upsamplers (keep for backward-compat; not needed if r is inferred in CA)
        self.hsi_upsamplers = nn.ModuleDict()
        for s in self.fusion_stages:
            sf = scaling_factors[self.fusion_stages.index(s)]
            if sf != 1.0:
                self.hsi_upsamplers[f'stage_{s}'] = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)

        # Adapters to decoder first channel
        dec_first = main_config['head'].get('decoder_channels', [256, 128, 64])[0]
        self.fusion_adapters = nn.ModuleDict({f'stage_{s}': nn.Conv2d(self.stage_dims[s], dec_first, 1) for s in self.fusion_stages})
        self.rgb_adapters = nn.ModuleDict({f'stage_{s}': nn.Conv2d(self.stage_dims[s], dec_first, 1) for s in self.rgb_only_stages})

        self.head = FusedFeatureFusionHead(main_config['head'], num_classes, self.all_stages)

    def forward(self, inputs):
        rgb_input = inputs['rgb']
        hsi_input = inputs['hsi']

        # Lists of per-stage features
        rgb_feats = self.rgb_model.backbone_forward(rgb_input)   # [(B,C,H,W), ...]
        hsi_feats = self.hsi_model.backbone_forward(hsi_input)   # [(B,C,H,W) or (B,C,S,H,W), ...]

        fused_stage_feats = {}  # change to dict
        for stage_idx in self.all_stages:
            rgb_feat = rgb_feats[stage_idx]
            hsi_feat = hsi_feats[stage_idx]

            if stage_idx in self.fusion_stages:
                # Promote 4D HSI features to 5D (S=1) for RGB-adapted HSI
                if hsi_feat.ndim == 4:
                    hsi_feat = hsi_feat.unsqueeze(2)  # (B, C, 1, H, W)
                elif hsi_feat.ndim != 5:
                    raise AssertionError(f"Expected HSI features to be 4D or 5D, got {hsi_feat.ndim}D at stage {stage_idx}")

                # Optional pre-CA upsample; keep spectral dim S
                up_key = f'stage_{stage_idx}'
                if up_key in self.hsi_upsamplers:
                    B, C, S, H, W = hsi_feat.shape
                    x = hsi_feat.reshape(B, C * S, H, W)
                    x = self.hsi_upsamplers[up_key](x)  # bilinear upsample
                    H_up, W_up = x.shape[-2], x.shape[-1]
                    hsi_feat = x.reshape(B, C, S, H_up, W_up)

                # Cross-attention fusion (returns (B, C, H_rgb, W_rgb))
                fused = self.cross_attention_modules[f'stage_{stage_idx}'](rgb_feat, hsi_feat)
                fused = self.fusion_adapters[f'stage_{stage_idx}'](fused)  # map to decoder first channel
                fused_stage_feats[f'stage_{stage_idx}'] = fused
            else:
                # No fusion: pass RGB through adapter
                adapted = self.rgb_adapters[f'stage_{stage_idx}'](rgb_feat)
                fused_stage_feats[f'stage_{stage_idx}'] = adapted

        logits = self.head(fused_stage_feats)
        if logits.shape[2:] != rgb_input.shape[2:]:
            logits = F.interpolate(logits, size=rgb_input.shape[2:], mode='bilinear', align_corners=False)
        return logits

def _initialize_model_weights(module, init_config):
    pretrained = init_config.get('pretrained', {})
    if not pretrained or pretrained.get('use_pretrained', "None").lower() == "none":
        print("No pretrained weights requested (use_pretrained=None).")
        return module, None

    modality = init_config.get('modality', 'rgb')
    # --- NEW: override modality for rgb_adapter path ---
    if modality == 'hsi' and getattr(getattr(module, 'backbone', None), 'hsi_mode', '') == 'rgb_adapter':
        print("Info: rgb_adapter path -> treating as RGB for pretrained weight loading.")
        modality = 'rgb'

    use_val = pretrained.get('use_pretrained', "").lower()
    if use_val != 'imagenet':
        print(f"No ImageNet transfer requested (use_pretrained='{use_val}').")
        return module, None

    model_name = pretrained.get('model_name', 'swin_tiny_patch4_window7_224')
    timm_state = None
    loaded_from = None

    # Try timm first (preferred for Swin)
    if TIMM_AVAILABLE:
        try:
            print(f"Attempting to load ImageNet pretrained weights via timm: {model_name}")
            timm_model = timm.create_model(model_name, pretrained=True)
            timm_state = timm_model.state_dict()
            loaded_from = 'timm'
        except Exception as e:
            print(f"Failed to load timm pretrained weights ({model_name}): {e}")

    # If timm failed (or model looks like SegFormer), try HF fallback
    if timm_state is None:
        look_like_segformer = (
            'mit' in model_name.lower()
            or 'segformer' in model_name.lower()
            or init_config.get('backbone', {}).get('type', '').lower() == 'segformer'
            or model_name.lower().startswith('nvidia/')
        )
        if look_like_segformer:
            try:
                from transformers import AutoModel
                hf_candidates = [model_name]
                if not model_name.startswith(('nvidia/', 'facebook/', 'Intel/', 'open-mmlab/')):
                    hf_candidates.append(f"nvidia/{model_name}")
                loaded = False
                for hf_name in hf_candidates:
                    try:
                        print(f"Attempting to load pretrained weights from HuggingFace model hub (safetensors preferred): {hf_name}")
                        hf_model = AutoModel.from_pretrained(hf_name, use_safetensors=True)
                        timm_state = hf_model.state_dict()
                        loaded_from = 'hf_safetensors'
                        print(f"Loaded HF safetensors for {hf_name}")
                        loaded = True
                        break
                    except Exception as e_s:
                        print(f"Safetensors load failed for {hf_name}: {e_s} — trying default HF load as fallback.")
                        try:
                            hf_model = AutoModel.from_pretrained(hf_name)
                            timm_state = hf_model.state_dict()
                            loaded_from = 'hf'
                            print(f"Loaded HF weights for {hf_name} (non-safetensors fallback)")
                            loaded = True
                            break
                        except Exception as e2:
                            print(f"Failed to load HuggingFace weights ({hf_name}): {e2}")
                if not loaded:
                    print("No HuggingFace weights found or failed to load for any candidate.")
            except ImportError:
                print("transformers not installed; skipping HF fallback.")

    if timm_state is None:
        print(f"No ImageNet pretrained weights available for {model_name}. Skipping ImageNet init.")
        return module, None

    # --- Decide transfer strategy based on source keys and target module keys ---
    src_keys = set(timm_state.keys())
    tgt_sd = module.state_dict()
    tgt_keys = set(tgt_sd.keys())

    print(f"[DEBUG] loaded_from={loaded_from} source_keys={len(src_keys)} target_keys={len(tgt_keys)}")
    sample_src = list(src_keys)[:30]
    sample_tgt = list(tgt_keys)[:30]
    print(f"[DEBUG] sample source keys (up to 30): {sample_src}")
    print(f"[DEBUG] sample target keys (up to 30): {sample_tgt}")

    # heuristics
    src_looks_hf_encoder = any(k.startswith('encoder.') for k in src_keys)
    tgt_has_backbone_hf_model = any(k.startswith('backbone.hf_model.encoder') for k in tgt_keys)
    direct_intersection = src_keys & tgt_keys

    info = None
    try:
        if direct_intersection:
            # direct match present: best for timm->swin direct transfer
            print(f"[DEBUG] direct intersection between source and target keys: {len(direct_intersection)}; using direct transfer.")
            # prefer direct exact matching transfer function
            if modality == 'rgb':
                info = _load_pretrained_weights_for_rgb(module, timm_state, init_config.get('backbone', {}))
            else:
                info = _load_pretrained_weights_for_hsi(module, timm_state, init_config.get('backbone', {}))

        elif src_looks_hf_encoder and tgt_has_backbone_hf_model:
            # HF segformer encoder keys need prefixed variants -> we augment and run rgb loader
            print("[DEBUG] source looks like HF SegFormer encoder and target expects backbone.hf_model.* -> using HF-aware augmentation transfer.")
            aug = _augment_source_keys(timm_state)
            if modality == 'rgb':
                info = _load_pretrained_weights_for_rgb(module, aug, init_config.get('backbone', {}))
            else:
                info = _load_pretrained_weights_for_hsi(module, aug, init_config.get('backbone', {}))

        else:
            # fallback: try augmented source keys (adds prefixed/stripped variants) then robust transfer
            print("[DEBUG] No direct intersection. Trying augmented source keys and robust heuristics.")
            aug = _augment_source_keys(timm_state)
            if modality == 'rgb':
                info = _load_pretrained_weights_for_rgb(module, aug, init_config.get('backbone', {}))
            else:
                info = _load_pretrained_weights_for_hsi(module, aug, init_config.get('backbone', {}))

        print(f"Successfully applied pretrained weights from: {model_name} (modality={modality})")
        return module, info
    except Exception as e:
        print(f"Failed while applying pretrained weights to module: {e}")
        return module, None


def _setup_fusion_parameter_groups(model, config):
    """Parameter groups for fusion models."""
    training = config.get('training', {})
    base_lr = training.get('learning_rate', 1e-4)
    lr_factors = training.get('lr_factors', {})
    head_lr = base_lr * lr_factors.get('head', 1.0)
    backbone_pretrained_lr = base_lr * lr_factors.get('backbone_pretrained', 0.1)

    model_type = config.get('model_type', 'logitfusion')
    g0, g1 = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if model_type == 'logitfusion':
            (g1 if 'fusion_layer' in name else g0).append(p)
        else:  # featurefusion
            if any(k in name for k in ['cross_attention', 'fusion_adapters', 'hsi_upsamplers', 'head']):
                g1.append(p)
            else:
                g0.append(p)

    model.param_groups = []
    if g0:
        model.param_groups.append({'params': g0, 'lr': backbone_pretrained_lr, 'name': 'backbone'})
    if g1:
        # give a clearer name depending on fusion type
        fusion_name = 'fusion_layer' if model_type == 'logitfusion' else 'fusion_and_head'
        model.param_groups.append({'params': g1, 'lr': head_lr, 'name': fusion_name})


def build_model_finetune(config, num_classes):
    """Build model by type and set parameter groups."""
    device_id = config['hardware']['gpu']
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model_type = config.get('model_type', 'unimodal')

    if model_type == 'unimodal':
        model_cfg = config.copy()
        pca_enabled = config.get('augmentation', {}).get('pca_transform', {}).get('enable', False)
        model_cfg.setdefault('backbone', {})
        model_cfg['backbone']['pca_enabled'] = pca_enabled

        model = UnimodalSegmentationModel(model_cfg, num_classes)
        model, transfer_info = _initialize_model_weights(model, model_cfg)
        _setup_model_parameter_groups(model, model_cfg, transfer_info)

    elif model_type == 'logitfusion':
        model = LateLogitFusionModel(config, num_classes, device)
        _setup_fusion_parameter_groups(model, config)

    elif model_type == 'featurefusion':
        model = CrossAttentionFusionModel(config, num_classes, device)
        _setup_fusion_parameter_groups(model, config)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Single concise print summarizing parameter groups and assigned learning rates
    if hasattr(model, 'param_groups') and model.param_groups:
        parts = []
        for g in model.param_groups:
            name = g.get('name', 'group')
            lr = g.get('lr', None)
            try:
                cnt = sum(p.numel() for p in g['params'] if getattr(p, 'requires_grad', True))
            except Exception:
                cnt = sum(getattr(p, 'numel', lambda: 0)() for p in g.get('params', []))
            parts.append(f"{name}: {cnt:,} params, LR={lr:.2e}" if lr is not None else f"{name}: {cnt:,} params")
        print("Parameter group assignment -> " + " | ".join(parts))

    model.to(device)
    return model


def _setup_model_parameter_groups(model, config, transfer_info):
    training = config.get('training', {})
    base_lr = training.get('learning_rate', 1e-4)
    lr_factors = training.get('lr_factors', {})
    head_lr = base_lr * lr_factors.get('head', 1.0)
    backbone_pretrained_lr = base_lr * lr_factors.get('backbone_pretrained', 0.1)
    backbone_random_lr = base_lr * lr_factors.get('backbone_random', 1.0)

    modality = config.get('modality', 'rgb')

    # --- NEW: force RGB grouping for rgb_adapter ---
    if modality == 'hsi' and getattr(getattr(model, 'backbone', None), 'hsi_mode', '') == 'rgb_adapter' and transfer_info:
        print("Info: rgb_adapter path -> using RGB parameter grouping (adapter params counted as random_backbone).")
        model.param_groups = _create_rgb_parameter_groups(
            model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr
        )
        return

    if modality == 'hsi' and transfer_info:
        model.param_groups = _create_hsi_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr)
    elif modality == 'rgb' and transfer_info:
        model.param_groups = _create_rgb_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr)
    else:
        # fallback (define backbone_lr explicitly)
        backbone_lr = base_lr * lr_factors.get('backbone_pretrained', 0.1)
        model.param_groups = _create_standard_parameter_groups(model, config, head_lr, backbone_lr)


def _create_hsi_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr):
    """Create optimizer groups for HSI model. Map transferred state-dict keys to model parameter names.

    Heuristics:
      1) exact match
      2) stripped-prefix match (remove common prefixes like 'backbone.' / 'backbone.hf_model.')
      3) add common prefixes to source key (in case source lacked them)
      4) longest common suffix (requires >=2 tokens)
    """
    transferred_keys = set(transfer_info.get('transferred_params', []))
    model_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    model_param_names = [n for n, _ in model_params]

    mapped_param_names = set()
    mapping_examples = []

    prefixes = ['backbone.hf_model.', 'backbone.', 'hf_model.', 'model.']

    for tkey in transferred_keys:
        # 1) exact
        if tkey in model_param_names:
            mapped_param_names.add(tkey)
            mapping_examples.append((tkey, tkey))
            continue

        # 2) stripped variants
        stripped = tkey
        for p in prefixes:
            if tkey.startswith(p):
                stripped = tkey[len(p):]
                break
        if stripped in model_param_names:
            mapped_param_names.add(stripped)
            mapping_examples.append((tkey, stripped))
            continue

        # 3) add prefixes (source lacked prefix but model has it)
        found = False
        for p in prefixes:
            cand = p + tkey
            if cand in model_param_names:
                mapped_param_names.add(cand)
                mapping_examples.append((cand, tkey))
                found = True
                break
        if found:
            continue

        # 4) suffix heuristic
        best, common_len = _best_suffix_match(tkey, model_param_names)
        if best and common_len >= 2:
            mapped_param_names.add(best)
            mapping_examples.append((best, tkey))
            continue

    # Build groups
    g_pretrained, g_random, g_head = [], [], []
    for name, p in model_params:
        if 'seg_head' in name:
            g_head.append(p)
        elif name in mapped_param_names:
            g_pretrained.append(p)
        else:
            g_random.append(p)

    # Debug prints
    try:
        print(f"_create_hsi_parameter_groups: transferred_keys={len(transferred_keys)}, mapped_params={len(mapped_param_names)}")
        if mapping_examples:
            print("  Examples mapping (source_key -> model_param):")
            for src, tgt in mapping_examples[:20]:
                print(f"    - {src}  ->  {tgt}")
        unmapped = sorted(list(transferred_keys - set(src for src, _ in mapping_examples)))
        if unmapped:
            print(f"  Transferred keys not mapped (examples up to 10): {unmapped[:10]}")
        def count_params(lst): return sum(p.numel() for p in lst)
        print(f"  Group sizes -> pretrained_backbone: {count_params(g_pretrained):,}, random_backbone: {count_params(g_random):,}, head: {count_params(g_head):,}")
        print(f"  LRs -> pretrained_backbone: {backbone_pretrained_lr:.2e}, random_backbone: {backbone_random_lr:.2e}, head: {head_lr:.2e}")
    except Exception as e:
        print(f"_create_hsi_parameter_groups debug failed: {e}")

    groups = []
    if g_pretrained:
        groups.append({'params': g_pretrained, 'lr': backbone_pretrained_lr, 'name': 'pretrained_backbone'})
    if g_random:
        groups.append({'params': g_random, 'lr': backbone_random_lr, 'name': 'random_backbone'})
    if g_head:
        groups.append({'params': g_head, 'lr': head_lr, 'name': 'head'})
    return groups


def _create_rgb_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr):
    transferred = set(transfer_info.get('transferred_params', []))
    g0, g1, g2 = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'seg_head' in name:
            g2.append(p)
        elif name in transferred:
            g0.append(p)
        else:
            g1.append(p)
    return [
        {'params': g0, 'lr': backbone_pretrained_lr, 'name': 'pretrained_backbone'},
        {'params': g1, 'lr': backbone_random_lr, 'name': 'random_backbone'},
        {'params': g2, 'lr': head_lr, 'name': 'head'},
    ]


def _create_standard_parameter_groups(model, config, head_lr, backbone_lr):
    pretrained = config.get('pretrained', {})
    use_pretrained = pretrained.get('use_pretrained', "None").lower() != "none"
    if use_pretrained:
        backbone_lr, head_lr = 1e-5, 1e-4

    g0, g1 = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (g1 if 'seg_head' in name else g0).append(p)
    return [
        {'params': g0, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': g1, 'lr': head_lr, 'name': 'head'},
    ]


def load_model_from_checkpoint(config_path, num_classes_from_dataset=None, checkpoint_path=None):
    """Build a model from config and load weights from a checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config.get('num_classes', num_classes_from_dataset)
    if num_classes is None:
        raise ValueError("num_classes must be provided.")
    config['num_classes'] = num_classes

    model = build_model_finetune(config, num_classes)

    if checkpoint_path is None:
        checkpoint_path = config.get('pretrained', {}).get('checkpoint_path')
    if not checkpoint_path:
        warnings.warn("No checkpoint path provided. Returning model with initial weights.")
        return model
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    cleaned = OrderedDict((k[10:] if k.startswith('_orig_mod.') else k, v) for k, v in state_dict.items())
    model.load_state_dict(cleaned, strict=True)
    print(f"Loaded model weights from checkpoint: {checkpoint_path}")
    return model


def _is_spectral_block_parameter(param_name):
    """True if parameter belongs to a spectral block (every 3rd block)."""
    if 'spectral_attn' in param_name:
        return True
    parts = param_name.split('.')
    if len(parts) >= 4 and parts[0] == 'layers' and parts[2] == 'blocks':
        try:
            return int(parts[3]) % 3 == 2
        except Exception:
            return False
    return False


def _load_pretrained_weights_for_hsi(hsi_model, timm_state_dict, model_config_params):
    """Transfer compatible RGB timm weights into HSI model, skipping spectral blocks."""
    hsi_sd = hsi_model.state_dict()
    transferred, shape_mismatch, not_found = [], [], []

    for hsi_name, hsi_val in hsi_sd.items():
        if any(k in hsi_name for k in ['seg_head', 'spectral_attn', 'relative_position_index']):
            continue
        if _is_spectral_block_parameter(hsi_name):
            continue

        rgb_name = hsi_name.replace('norm_ffn', 'norm2') if 'norm_ffn' in hsi_name else hsi_name
        if rgb_name in timm_state_dict:
            rgb_val = timm_state_dict[rgb_name]
            if hsi_val.shape == rgb_val.shape:
                hsi_sd[hsi_name] = rgb_val.clone()
                transferred.append(hsi_name)
            else:
                shape_mismatch.append((hsi_name, tuple(hsi_val.shape), tuple(rgb_val.shape)))
        else:
            not_found.append(hsi_name)

    # Optional smart init for extra spatial blocks (example mapping kept minimal)
    smart_map = [
        ('layers.2.blocks.6', 'layers.2.blocks.4'),
        ('layers.2.blocks.7', 'layers.2.blocks.5'),
    ]
    components = [
        'norm1.weight', 'norm1.bias',
        'attn.relative_position_bias_table',
        'attn.qkv.weight', 'attn.qkv.bias',
        'attn.proj.weight', 'attn.proj.bias',
        'norm2.weight', 'norm2.bias',
        'mlp.fc1.weight', 'mlp.fc1.bias',
        'mlp.fc2.weight', 'mlp.fc2.bias',
    ]
    for tgt_block, src_block in smart_map:
        for comp in components:
            tgt = f"{tgt_block}.{comp}"
            src = f"{src_block}.{comp}"
            if tgt in hsi_sd and src in timm_state_dict and tgt not in transferred:
                if hsi_sd[tgt].shape == timm_state_dict[src].shape:
                    hsi_sd[tgt] = timm_state_dict[src].clone()
                    transferred.append(tgt)

    hsi_model.load_state_dict(hsi_sd)

    return {
        'transferred_params': transferred,
        'shape_mismatches': shape_mismatch,
        'not_found_in_timm': not_found,
    }


def _best_suffix_match(target_key, candidate_keys):
    """Return (best_key, common_token_count) with longest common suffix (dot-separated)."""
    tgt_tokens = target_key.split('.')
    best = None
    best_len = 0
    for ck in candidate_keys:
        ck_tokens = ck.split('.')
        # count common suffix tokens
        common = 0
        for i in range(1, min(len(tgt_tokens), len(ck_tokens)) + 1):
            if tgt_tokens[-i] == ck_tokens[-i]:
                common += 1
            else:
                break
        if common > best_len:
            best_len = common
            best = ck
    return best, best_len


def _augment_source_keys(state_dict):
    """Create prefixed and stripped variants for source keys to improve exact matching."""
    prefixes = ['', 'backbone.hf_model.', 'backbone.', 'hf_model.', 'model.']
    aug = {}
    for k, v in state_dict.items():
        # keep original
        if k not in aug:
            aug[k] = v
        # add prefixed variants
        for p in prefixes:
            nk = p + k
            if nk not in aug:
                aug[nk] = v
        # add stripped variants if key starts with a known prefix
        for p in prefixes[1:]:
            if k.startswith(p):
                nk = k[len(p):]
                if nk not in aug:
                    aug[nk] = v
    return aug


def _load_pretrained_weights_for_rgb(rgb_model, timm_state_dict, model_config_params):
    """Robust transfer of HF/timm weights into the RGB model with debug prints."""
    sd = rgb_model.state_dict()
    transferred, shape_mismatch, not_found = [], [], []
    mapping = {}  # target_name -> source_key used

    # create augmented source key space (adds prefixed / stripped variants)
    aug_source = _augment_source_keys(timm_state_dict)
    source_keys = list(aug_source.keys())

    # prefer exact matches (including augmented variants), then best-suffix heuristic
    for name, val in sd.items():
        if any(k in name for k in ['seg_head', 'relative_position_index']):
            continue

        # 1) direct/augmented exact match
        if name in aug_source:
            s_val = aug_source[name]
            if tuple(val.shape) == tuple(s_val.shape):
                sd[name] = s_val.clone()
                transferred.append(name)
                mapping[name] = name
                continue
            else:
                shape_mismatch.append((name, tuple(val.shape), tuple(s_val.shape)))
                continue

        # 2) try stripping common prefixes from target and match again
        stripped = name
        for p in ['backbone.hf_model.', 'backbone.', 'hf_model.', 'model.']:
            if name.startswith(p):
                stripped = name[len(p):]
                break
        if stripped in aug_source:
            s_val = aug_source[stripped]
            if tuple(val.shape) == tuple(s_val.shape):
                sd[name] = s_val.clone()
                transferred.append(name)
                mapping[name] = stripped
                continue
            else:
                shape_mismatch.append((name, tuple(val.shape), tuple(s_val.shape)))
                continue

        # 3) fallback: best suffix match (require >=2 tokens in common)
        best_key, common_len = _best_suffix_match(name, source_keys)
        if best_key and common_len >= 2:
            s_val = aug_source[best_key]
            if tuple(val.shape) == tuple(s_val.shape):
                sd[name] = s_val.clone()
                transferred.append(name)
                mapping[name] = best_key
                continue
            else:
                shape_mismatch.append((name, tuple(val.shape), tuple(s_val.shape)))
                continue

        # 4) not found
        not_found.append(name)

    # apply state dict and print concise debug info
    rgb_model.load_state_dict(sd)

    print(f"\n_load_pretrained_weights_for_rgb: transferred={len(transferred)}, shape_mismatch={len(shape_mismatch)}, not_found={len(not_found)}")
    if transferred:
        print("  Examples of transferred (target_name <- source_key):")
        for t in transferred[:20]:
            print(f"    - {t}  <-  {mapping.get(t, '<unknown>')}  shape={tuple(rgb_model.state_dict()[t].shape)}")
    if shape_mismatch:
        print("  Examples of shape mismatches (target, target_shape, source_shape):")
        for mm in shape_mismatch[:10]:
            print(f"    - {mm}")
    if not_found:
        print("  Examples of target names not found in source (first 20):")
        for nm in not_found[:20]:
            print(f"    - {nm}")

    # also show a few sample source keys to inspect naming
    sample_src = list(timm_state_dict.keys())[:30]
    print(f"  Sample original source keys (up to 30): {sample_src}")

    return {
        'transferred_params': transferred,
        'shape_mismatches': shape_mismatch,
        'not_found_in_timm': not_found,
        'mapping': mapping,
    }