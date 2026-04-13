from collections import OrderedDict
from pathlib import Path
import warnings
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from .backbones import build_swin_transformer
from .segmentation_head import build_segmentation_head, FusedFeatureFusionHead
from .cross_attention_fusion import BidirectionalCrossAttention


class UnimodalSegmentationModel(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        if num_classes is not None:
            self.config['num_classes'] = num_classes

        self.backbone = build_swin_transformer(config)
        self.num_features = getattr(self.backbone, 'num_features', None)
        if self.num_features is None:
            if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone, 'embed_dim'):
                self.num_features = int(self.backbone.embed_dim * 2 ** (getattr(self.backbone, 'num_layers', 4) - 1))
            else:
                raise RuntimeError("Backbone did not expose 'num_features' attribute; cannot construct head.")

        if hasattr(self.backbone, 'out_channels'):
            try:
                config.setdefault('backbone', {})['out_channels'] = list(self.backbone.out_channels)
            except Exception:
                config.setdefault('backbone', {})['out_channels'] = getattr(self.backbone, 'out_channels')

        head_cfg = config
        if self.config.get('modality') == 'hsi' and not hasattr(self.backbone, 'patch_embed'):
            head_cfg = config.copy()
            head_cfg['modality'] = 'rgb'
        elif self.config.get('modality') == 'hsi' and getattr(self.backbone, 'hsi_mode', '') == 'rgb_adapter':
            head_cfg = config.copy()
            head_cfg['modality'] = 'rgb'

        self.seg_head = build_segmentation_head(head_cfg, self.num_features)

        register_legacy_aliases = config.get('backbone', {}).get('register_legacy_aliases', False)

        if register_legacy_aliases and hasattr(self.backbone, 'patch_embed'):
            self.patch_embed = self.backbone.patch_embed
        if register_legacy_aliases and hasattr(self.backbone, 'pos_drop'):
            self.pos_drop = self.backbone.pos_drop
        if register_legacy_aliases and hasattr(self.backbone, 'layers'):
            self.layers = self.backbone.layers
        if register_legacy_aliases and hasattr(self.backbone, 'norm'):
            self.norm = self.backbone.norm

    def forward_features(self, x):
        if getattr(self.backbone, 'hsi_mode', '') == 'rgb_adapter':
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError("HSI input must be 4D Tensor [B,C,H,W].")
            x = self.backbone.adapter(x)
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
        return self.forward_features(x)


def _load_unimodal_component(config_path, checkpoint_path, num_classes, device, name="component"):
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)

    cfg['num_classes'] = num_classes
    cfg['pretrained'] = {'use_pretrained': "None"}

    model = UnimodalSegmentationModel(cfg)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

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
            missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = model.load_state_dict(sd, strict=False)
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
        fusion_direction = fusion_cfg.get('fusion_direction', 'bidirectional')

        base_dim = self.rgb_model.config.get('embed_dim', 96)
        self.stage_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.all_stages = [0, 1, 2, 3]
        self.rgb_only_stages = [s for s in self.all_stages if s not in self.fusion_stages]

        with open(main_config['hsi_model_config_path'], 'r', encoding='utf-8-sig') as f:
            hsi_cfg = yaml.safe_load(f)
        hsi_heads = hsi_cfg.get('backbone', {}).get('num_heads', [3, 6, 12, 24])

        self.cross_attention_modules = nn.ModuleDict()
        for s in self.fusion_stages:
            heads = hsi_heads[s] if s < len(hsi_heads) else hsi_heads[-1]
            self.cross_attention_modules[f'stage_{s}'] = BidirectionalCrossAttention(
                d_model=self.stage_dims[s],
                num_heads=heads,
                num_cross_attention_layers=num_ca_layers,
                fusion_direction=fusion_direction
            )

        self.hsi_upsamplers = nn.ModuleDict()
        for s in self.fusion_stages:
            sf = scaling_factors[self.fusion_stages.index(s)]
            if sf != 1.0:
                self.hsi_upsamplers[f'stage_{s}'] = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)

        dec_first = main_config['head'].get('decoder_channels', [256, 128, 64])[0]
        self.fusion_adapters = nn.ModuleDict({f'stage_{s}': nn.Conv2d(self.stage_dims[s], dec_first, 1) for s in self.fusion_stages})
        self.rgb_adapters = nn.ModuleDict({f'stage_{s}': nn.Conv2d(self.stage_dims[s], dec_first, 1) for s in self.rgb_only_stages})

        self.head = FusedFeatureFusionHead(main_config['head'], num_classes, self.all_stages)

    def forward(self, inputs):
        rgb_input = inputs['rgb']
        hsi_input = inputs['hsi']

        rgb_feats = self.rgb_model.backbone_forward(rgb_input)
        hsi_feats = self.hsi_model.backbone_forward(hsi_input)

        fused_stage_feats = {}
        for stage_idx in self.all_stages:
            rgb_feat = rgb_feats[stage_idx]
            hsi_feat = hsi_feats[stage_idx]

            if stage_idx in self.fusion_stages:
                if hsi_feat.ndim == 4:
                    hsi_feat = hsi_feat.unsqueeze(2)
                elif hsi_feat.ndim != 5:
                    raise AssertionError(f"Expected HSI features to be 4D or 5D, got {hsi_feat.ndim}D at stage {stage_idx}")

                up_key = f'stage_{stage_idx}'
                if up_key in self.hsi_upsamplers:
                    B, C, S, H, W = hsi_feat.shape
                    x = hsi_feat.reshape(B, C * S, H, W)
                    x = self.hsi_upsamplers[up_key](x)
                    H_up, W_up = x.shape[-2], x.shape[-1]
                    hsi_feat = x.reshape(B, C, S, H_up, W_up)

                fused = self.cross_attention_modules[f'stage_{stage_idx}'](rgb_feat, hsi_feat)
                fused = self.fusion_adapters[f'stage_{stage_idx}'](fused)
                fused_stage_feats[f'stage_{stage_idx}'] = fused
            else:
                adapted = self.rgb_adapters[f'stage_{stage_idx}'](rgb_feat)
                fused_stage_feats[f'stage_{stage_idx}'] = adapted

        logits = self.head(fused_stage_feats)
        if logits.shape[2:] != rgb_input.shape[2:]:
            logits = F.interpolate(logits, size=rgb_input.shape[2:], mode='bilinear', align_corners=False)
        return logits


def _initialize_model_weights(module, init_config):
    pretrained = init_config.get('pretrained', {})
    if not pretrained or pretrained.get('use_pretrained', "None").lower() == "none":
        return module, None

    modality = init_config.get('modality', 'rgb')
    if modality == 'hsi' and getattr(getattr(module, 'backbone', None), 'hsi_mode', '') == 'rgb_adapter':
        modality = 'rgb'

    use_val = pretrained.get('use_pretrained', "").lower()
    if use_val != 'imagenet':
        return module, None

    model_name = pretrained.get('model_name', 'swin_tiny_patch4_window7_224')
    timm_state = None

    if TIMM_AVAILABLE:
        try:
            timm_model = timm.create_model(model_name, pretrained=True)
            timm_state = timm_model.state_dict()
        except Exception as e:
            print(f"Warning: timm model load failed: {e}")
    else:
        print(f"Warning: use_pretrained='ImageNet' configured but timm is not available. Skipping pretrained weight loading.")

    if timm_state is None:
        return module, None

    info = None
    try:
        if modality == 'rgb':
            info = _load_pretrained_weights_for_rgb(module, timm_state, init_config.get('backbone', {}))
        else:
            info = _load_pretrained_weights_for_hsi(module, timm_state, init_config.get('backbone', {}))
    except Exception as e:
        print(f"Warning: pretrained weight loading failed: {e}")
        return module, None

    print(f"Pretrained weights loaded ({model_name}): {len(info.get('transferred_params', []))} transferred")
    return module, info


def _setup_fusion_parameter_groups(model, config):
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
        else:
            if any(k in name for k in ['cross_attention', 'fusion_adapters', 'hsi_upsamplers', 'head']):
                g1.append(p)
            else:
                g0.append(p)

    model.param_groups = []
    if g0:
        model.param_groups.append({'params': g0, 'lr': backbone_pretrained_lr, 'name': 'backbone'})
    if g1:
        fusion_name = 'fusion_layer' if model_type == 'logitfusion' else 'fusion_and_head'
        model.param_groups.append({'params': g1, 'lr': head_lr, 'name': fusion_name})


def build_model_finetune(config, num_classes):
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

    if modality == 'hsi' and getattr(getattr(model, 'backbone', None), 'hsi_mode', '') == 'rgb_adapter' and transfer_info:
        model.param_groups = _create_rgb_parameter_groups(
            model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr
        )
        return

    if modality == 'hsi' and transfer_info:
        model.param_groups = _create_hsi_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr)
    elif modality == 'rgb' and transfer_info:
        model.param_groups = _create_rgb_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr)
    else:
        backbone_lr = base_lr * lr_factors.get('backbone_pretrained', 0.1)
        model.param_groups = _create_standard_parameter_groups(model, config, head_lr, backbone_lr)


def _create_hsi_parameter_groups(model, transfer_info, head_lr, backbone_pretrained_lr, backbone_random_lr):
    transferred_keys = set(transfer_info.get('transferred_params', []))
    model_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    model_param_names = [n for n, _ in model_params]

    mapped_param_names = set()
    prefixes = ['backbone.hf_model.', 'backbone.', 'hf_model.', 'model.']

    for tkey in transferred_keys:
        if tkey in model_param_names:
            mapped_param_names.add(tkey)
            continue

        stripped = tkey
        for p in prefixes:
            if tkey.startswith(p):
                stripped = tkey[len(p):]
                break
        if stripped in model_param_names:
            mapped_param_names.add(stripped)
            continue

        found = False
        for p in prefixes:
            cand = p + tkey
            if cand in model_param_names:
                mapped_param_names.add(cand)
                found = True
                break
        if found:
            continue

        best, common_len = _best_suffix_match(tkey, model_param_names)
        if best and common_len >= 2:
            mapped_param_names.add(best)
            continue

    g_pretrained, g_random, g_head = [], [], []
    for name, p in model_params:
        if 'seg_head' in name:
            g_head.append(p)
        elif name in mapped_param_names:
            g_pretrained.append(p)
        else:
            g_random.append(p)

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
    with open(config_path, 'r', encoding='utf-8-sig') as f:
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
    return model


def _is_spectral_block_parameter(param_name):
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
    tgt_tokens = target_key.split('.')
    best = None
    best_len = 0
    for ck in candidate_keys:
        ck_tokens = ck.split('.')
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
    prefixes = ['', 'backbone.hf_model.', 'backbone.', 'hf_model.', 'model.']
    aug = {}
    for k, v in state_dict.items():
        if k not in aug:
            aug[k] = v
        for p in prefixes:
            nk = p + k
            if nk not in aug:
                aug[nk] = v
        for p in prefixes[1:]:
            if k.startswith(p):
                nk = k[len(p):]
                if nk not in aug:
                    aug[nk] = v
    return aug


def _load_pretrained_weights_for_rgb(rgb_model, timm_state_dict, model_config_params):
    sd = rgb_model.state_dict()
    transferred, shape_mismatch, not_found = [], [], []
    mapping = {}

    aug_source = _augment_source_keys(timm_state_dict)
    source_keys = list(aug_source.keys())

    for name, val in sd.items():
        if any(k in name for k in ['seg_head', 'relative_position_index']):
            continue

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

        not_found.append(name)

    rgb_model.load_state_dict(sd)

    return {
        'transferred_params': transferred,
        'shape_mismatches': shape_mismatch,
        'not_found_in_timm': not_found,
        'mapping': mapping,
    }