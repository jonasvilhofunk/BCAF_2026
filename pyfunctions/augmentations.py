import random
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Augmentation:
    def _parse_size_config(self, size_val, cfg_name):
        if isinstance(size_val, int):
            if size_val > 0:
                return (size_val, size_val)
            print(f"Warning (Augmentation): {cfg_name} must be positive, got {size_val}.")
            return None
        if isinstance(size_val, (list, tuple)):
            if len(size_val) != 2:
                print(f"Warning (Augmentation): {cfg_name} must have 2 elements (H,W).")
                return None
            try:
                h, w = int(size_val[0]), int(size_val[1])
                if h > 0 and w > 0:
                    return (h, w)
                print(f"Warning (Augmentation): {cfg_name} must be positive, got {size_val}.")
                return None
            except ValueError:
                print(f"Warning (Augmentation): {cfg_name} must be integers, got {size_val}.")
                return None
        if size_val is None:
            return None
        print(f"Warning (Augmentation): {cfg_name} has invalid type {type(size_val)}.")
        return None

    def __init__(self, config, is_train=True, data_modality_to_load='rgb', hsi_stats_dict=None):
        self.config = config.get('augmentation', config)
        self.is_train = is_train
        self.data_modality_to_load = data_modality_to_load.lower()

        self.rgb_norm_config = self.config.get('rgb_normalization', {})
        self.rgb_norm_enable = self.rgb_norm_config.get('enable', False)
        self.rgb_norm_method = self.rgb_norm_config.get('method', 'none').lower()
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        self.rgb_cj_config = self.config.get('rgb_color_jitter', {})
        self.rgb_cj_enable = bool(self.rgb_cj_config.get('enable', False) and self.is_train)
        if self.rgb_cj_enable:
            p = float(self.rgb_cj_config.get('p', 0.8))
            b = float(self.rgb_cj_config.get('brightness', 0.2))
            c = float(self.rgb_cj_config.get('contrast', 0.2))
            s = float(self.rgb_cj_config.get('saturation', 0.2))
            h = float(self.rgb_cj_config.get('hue', 0.05))
            self.rgb_cj = T.RandomApply([T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)], p=p)
        else:
            self.rgb_cj = None

        self.hsi_norm_config = self.config.get('hsi_normalization', {})
        self.hsi_norm_enable = self.hsi_norm_config.get('enable', False)
        self.hsi_norm_method = self.hsi_norm_config.get('method', 'none').lower()
        self.hsi_stats = None
        self.hsi_stats_loaded_successfully = False

        if self.hsi_norm_enable and hsi_stats_dict and ('hsi' in self.data_modality_to_load):
            processed_stats = self._process_loaded_hsi_stats(hsi_stats_dict, self.hsi_norm_method)
            if processed_stats is not None:
                self.hsi_stats = processed_stats
                self.hsi_stats_loaded_successfully = True
            else:
                print("Warning (Augmentation): Invalid HSI stats. Normalization disabled.")
                self.hsi_norm_enable = False
        elif self.hsi_norm_enable and ('hsi' in self.data_modality_to_load) and not hsi_stats_dict:
            print("Warning (Augmentation): HSI normalization enabled but no stats provided. Disabled.")
            self.hsi_norm_enable = False

        self.hsi_spectral_padding_config = self.config.get('pad_channels', {})
        self.hsi_spectral_pad_enable = self.hsi_spectral_padding_config.get('enable', False)
        self.hsi_target_spectral_channels = self.hsi_spectral_padding_config.get('pad_channels_to', None)
        self.hsi_spectral_dim_axis = 0
        if self.hsi_spectral_pad_enable and self.hsi_target_spectral_channels is None and 'hsi' in self.data_modality_to_load:
            print("Warning (Augmentation): pad_channels enabled but 'pad_channels_to' missing. Disabled.")
            self.hsi_spectral_pad_enable = False

        self.resize_enable = False
        self.rgb_img_size = None
        self.hsi_img_size = None
        self.label_img_size = None

        if data_modality_to_load == 'rgb':
            resize_cfg = self.config.get('resize', {})
            self.resize_enable = resize_cfg.get('enable', False)
            if self.resize_enable:
                sz = self._parse_size_config(resize_cfg.get('size'), "resize.size (rgb)")
                if sz:
                    self.rgb_img_size = sz
                    self.label_img_size = sz
                else:
                    self.resize_enable = False
        elif data_modality_to_load == 'hsi':
            resize_cfg = self.config.get('resize', {})
            self.resize_enable = resize_cfg.get('enable', False)
            if self.resize_enable:
                sz = self._parse_size_config(resize_cfg.get('size'), "resize.size (hsi)")
                if sz:
                    self.hsi_img_size = sz
                    self.label_img_size = sz
                else:
                    self.resize_enable = False
        elif data_modality_to_load == 'rgb_hsi':
            resize_rgb = self.config.get('resize_rgb', {})
            resize_hsi = self.config.get('resize_hsi', {})
            en_rgb = resize_rgb.get('enable', False)
            en_hsi = resize_hsi.get('enable', False)
            self.resize_enable = en_rgb or en_hsi

            if en_rgb:
                self.rgb_img_size = self._parse_size_config(resize_rgb.get('size'), "resize_rgb.size")
                if not self.rgb_img_size:
                    en_rgb = False
            if en_hsi:
                self.hsi_img_size = self._parse_size_config(resize_hsi.get('size'), "resize_hsi.size")
                if not self.hsi_img_size:
                    en_hsi = False

            self.resize_enable = en_rgb or en_hsi
            if self.resize_enable:
                self.label_img_size = self.rgb_img_size or self.hsi_img_size
                if not self.label_img_size:
                    print("Warning (Augmentation): resize enabled for rgb_hsi but no target size set.")
                    self.resize_enable = False

        self.crop_enable = False
        self.crop_size_cfg_tuple = None
        self.crop_target_size_rgb = None
        self.crop_target_size_hsi = None
        self.crop_target_size_label = None

        if data_modality_to_load == 'rgb':
            crop_cfg = self.config.get('random_crop', {})
            self.crop_enable = crop_cfg.get('enable', False) and self.is_train
            if self.crop_enable:
                sz = self._parse_size_config(crop_cfg.get('size'), "random_crop.size (rgb)")
                if sz:
                    self.crop_size_cfg_tuple = sz
                    self.crop_target_size_rgb = sz
                    self.crop_target_size_label = sz
                else:
                    self.crop_enable = False
        elif data_modality_to_load == 'hsi':
            crop_cfg = self.config.get('random_crop', {})
            self.crop_enable = crop_cfg.get('enable', False) and self.is_train
            if self.crop_enable:
                sz = self._parse_size_config(crop_cfg.get('size'), "random_crop.size (hsi)")
                if sz:
                    self.crop_size_cfg_tuple = sz
                    self.crop_target_size_hsi = sz
                    self.crop_target_size_label = sz
                else:
                    self.crop_enable = False
        elif data_modality_to_load == 'rgb_hsi':
            crop_rgb = self.config.get('random_crop_rgb', {})
            crop_hsi = self.config.get('random_crop_hsi', {})
            crop_gen = self.config.get('random_crop', {})

            en_rgb = crop_rgb.get('enable', False) and self.is_train
            en_hsi = crop_hsi.get('enable', False) and self.is_train
            en_gen = crop_gen.get('enable', False) and self.is_train
            self.crop_enable = en_rgb or en_hsi or en_gen

            if self.crop_enable:
                s_rgb = self._parse_size_config(crop_rgb.get('size') if en_rgb else None, "random_crop_rgb.size") if en_rgb else None
                s_hsi = self._parse_size_config(crop_hsi.get('size') if en_hsi else None, "random_crop_hsi.size") if en_hsi else None
                s_gen = self._parse_size_config(crop_gen.get('size') if en_gen else None, "random_crop.size (rgb_hsi)") if en_gen else None

                common = None
                if s_rgb:
                    common = s_rgb
                    self.crop_target_size_rgb = s_rgb
                if s_hsi:
                    self.crop_target_size_hsi = s_hsi
                    if common is None:
                        common = s_hsi
                    elif common != s_hsi:
                        print(f"Warning (Augmentation): rgb/hsi crop sizes differ. Using RGB size {common}.")
                if common is None and s_gen:
                    common = s_gen
                    if en_rgb and self.crop_target_size_rgb is None:
                        self.crop_target_size_rgb = s_gen
                    if en_hsi and self.crop_target_size_hsi is None:
                        self.crop_target_size_hsi = s_gen

                if common:
                    self.crop_size_cfg_tuple = common
                    self.crop_target_size_label = common
                else:
                    print("Warning (Augmentation): Crop enabled but no valid size. Disabled.")
                    self.crop_enable = False

        self.rotate_config = self.config.get('random_rotate', {})
        self.rotate_enable = self.rotate_config.get('enable', False) and self.is_train
        self.rotate_degrees = self.rotate_config.get('degrees', [-10, 10])

        self.scale_config = self.config.get('random_scale', {})
        self.scale_enable = self.scale_config.get('enable', False) and self.is_train
        self.scale_limit = self.scale_config.get('scale_limit', [0.8, 1.2])

        self.flip_config = self.config.get('random_flip', {})
        self.flip_enable = self.flip_config.get('enable', False) and self.is_train
        self.p_horizontal = self.flip_config.get('p_horizontal', 0.5)
        self.p_vertical = self.flip_config.get('p_vertical', 0.0)

        self.hsi_sj_config = self.config.get('hsi_spectral_jitter', {})
        self.hsi_sj_enable = bool(self.hsi_sj_config.get('enable', False) and self.is_train)
        if self.hsi_sj_enable:
            self.hsi_sj_brightness = float(self.hsi_sj_config.get('brightness', 0.05))
            self.hsi_sj_contrast = float(self.hsi_sj_config.get('contrast', 0.05))

    def _process_loaded_hsi_stats(self, stats_dict, method):
        processed = {}
        try:
            if method == "minmax":
                mm = stats_dict.get('minmax', {})
                mins_list = mm.get('hsi_global_mins')
                maxs_list = mm.get('hsi_global_maxs')
                if not (isinstance(mins_list, list) and isinstance(maxs_list, list) and mins_list and len(mins_list) == len(maxs_list)):
                    print("Error (Augmentation): Invalid minmax stats.")
                    return None
                processed['mins'] = torch.tensor(mins_list, dtype=torch.float32)
                processed['maxs'] = torch.tensor(maxs_list, dtype=torch.float32)
                processed['ranges'] = processed['maxs'] - processed['mins']
                processed['ranges'][processed['ranges'] == 0] = 1.0e-6
            elif method == "standardize":
                st = stats_dict.get('standardize', {})
                means_list = st.get('hsi_global_means')
                stds_list = st.get('hsi_global_stds')
                if not (isinstance(means_list, list) and isinstance(stds_list, list) and means_list and len(means_list) == len(stds_list)):
                    print("Error (Augmentation): Invalid standardize stats.")
                    return None
                processed['means'] = torch.tensor(means_list, dtype=torch.float32)
                processed['stds'] = torch.tensor(stds_list, dtype=torch.float32)
                processed['stds'][processed['stds'] == 0] = 1.0e-6
            elif method == "none":
                return {}
            else:
                print(f"Warning (Augmentation): Unknown HSI norm method '{method}'.")
                return None
            return processed
        except Exception as e:
            print(f"Error (Augmentation): HSI stats processing failed: {e}")
            return None

    def _apply_normalization(self, img, img_type):
        if img is None:
            return None

        if img_type == 'rgb':
            if self.rgb_norm_enable and self.rgb_norm_method == 'imagenet':
                return TF.normalize(img, mean=self.imagenet_mean, std=self.imagenet_std)
            return img

        if img_type == 'hsi':
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"HSI image must be a torch.Tensor, got {type(img)}")
            hsi = img.float()

            if self.hsi_norm_enable and self.hsi_stats_loaded_successfully:
                K = hsi.shape[0]
                mask = (hsi != 0)
                if self.hsi_norm_method == 'standardize':
                    if 'means' in self.hsi_stats and 'stds' in self.hsi_stats:
                        k = min(K, self.hsi_stats['means'].shape[0], self.hsi_stats['stds'].shape[0])
                        if k <= 0:
                            return hsi
                        means = self.hsi_stats['means'][:k].view(-1, 1, 1).to(hsi.device)
                        stds = self.hsi_stats['stds'][:k].view(-1, 1, 1).to(hsi.device)
                        out = hsi.clone()
                        norm_vals = (hsi[:k] - means) / stds
                        out[:k][mask[:k]] = norm_vals[mask[:k]]
                        return out
                    print("Warning (Augmentation): Missing mean/std for HSI standardization. Skipped.")
            return hsi

        return img

    def _pad_hsi_spectral(self, hsi_tensor):
        if not isinstance(hsi_tensor, torch.Tensor):
            raise TypeError("HSI must be a Tensor for spectral padding.")

        cur_c = hsi_tensor.shape[self.hsi_spectral_dim_axis]
        tgt_c = self.hsi_target_spectral_channels

        if cur_c == tgt_c:
            return hsi_tensor
        if cur_c > tgt_c:
            idx = torch.arange(tgt_c, device=hsi_tensor.device)
            return torch.index_select(hsi_tensor, self.hsi_spectral_dim_axis, idx)

        pad_needed = tgt_c - cur_c
        pad_cfg = (0, 0, 0, 0, 0, pad_needed)
        return F.pad(hsi_tensor, pad_cfg, mode='constant', value=0)

    def _get_image_dims(self, img):
        if isinstance(img, Image.Image):
            return img.height, img.width
        if isinstance(img, torch.Tensor):
            return img.shape[-2], img.shape[-1]
        return -1, -1

    def _apply_single_geometric_transform(self, img, transform_name, param, is_label=False):
        if img is None:
            return None

        interp = T.InterpolationMode.NEAREST if is_label else T.InterpolationMode.BILINEAR
        fill_val = 0

        if transform_name == 'resize':
            h, w = self._get_image_dims(img)
            th, tw = param
            if h == th and w == tw:
                return img
            if interp == T.InterpolationMode.BILINEAR:
                return TF.resize(img, list(param), interpolation=interp, antialias=True)
            return TF.resize(img, list(param), interpolation=interp)

        if transform_name == 'hflip' and param:
            return TF.hflip(img)
        if transform_name == 'vflip' and param:
            return TF.vflip(img)
        if transform_name == 'rotate':
            return TF.rotate(img, param, interpolation=interp, fill=fill_val)
        if transform_name == 'crop':
            i, j, h, w = param
            return TF.crop(img, i, j, h, w)
        if transform_name == 'scale':
            h, w = self._get_image_dims(img)
            if h <= 0 or w <= 0:
                return img
            nh, nw = int(round(h * param)), int(round(w * param))
            if nh <= 0 or nw <= 0:
                return img
            if interp == T.InterpolationMode.BILINEAR:
                return TF.resize(img, [nh, nw], interpolation=interp, antialias=True)
            return TF.resize(img, [nh, nw], interpolation=interp)

        return img

    def _apply_geometric_transforms_sequentially(self, rgb_input, hsi_input, label_input_pil):
        rgb_tensor = None
        if rgb_input is not None:
            if isinstance(rgb_input, Image.Image):
                rgb_tensor = TF.to_tensor(rgb_input)
            elif isinstance(rgb_input, torch.Tensor):
                rgb_tensor = rgb_input.float()
                if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] in [1, 3]:
                    if torch.max(rgb_tensor) > 1.0:
                        rgb_tensor = rgb_tensor / 255.0
                elif rgb_tensor.ndim == 3 and rgb_tensor.shape[2] in [1, 3]:
                    if torch.max(rgb_tensor) > 1.0:
                        rgb_tensor = rgb_tensor / 255.0
                    rgb_tensor = rgb_tensor.permute(2, 0, 1)
                elif rgb_tensor.ndim == 2:
                    if torch.max(rgb_tensor) > 1.0:
                        rgb_tensor = rgb_tensor / 255.0
                    rgb_tensor = rgb_tensor.unsqueeze(0)
                else:
                    raise ValueError(f"RGB tensor has unexpected shape: {rgb_tensor.shape}")
                if rgb_tensor.shape[0] not in [1, 3]:
                    raise ValueError(f"RGB tensor must have 1 or 3 channels, got {rgb_tensor.shape[0]}")
            else:
                raise TypeError(f"RGB input must be PIL Image or Tensor, got {type(rgb_input)}")

        hsi_tensor = None
        if hsi_input is not None:
            if isinstance(hsi_input, np.ndarray):
                arr = hsi_input
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=2)
                if arr.ndim != 3:
                    raise ValueError(f"HSI array must be 3D (H,W,C) or 2D (H,W), got {arr.ndim}D")
                hsi_tensor = torch.from_numpy(arr.transpose((2, 0, 1))).float()
            elif isinstance(hsi_input, torch.Tensor):
                hsi_tensor = hsi_input.float()
                if hsi_tensor.ndim == 3 and hsi_tensor.shape[0] > hsi_tensor.shape[2] and hsi_tensor.shape[1] > hsi_tensor.shape[2] and hsi_tensor.shape[2] < 20:
                    hsi_tensor = hsi_tensor.permute(2, 0, 1)
                elif hsi_tensor.ndim == 2:
                    hsi_tensor = hsi_tensor.unsqueeze(0)
            else:
                raise TypeError(f"HSI input must be NumPy array or Tensor, got {type(hsi_input)}")

        current_label_pil = label_input_pil

        if hsi_tensor is not None:
            hsi_tensor = self._apply_spectral_jitter(hsi_tensor)

        current_rgb_img = rgb_tensor
        current_hsi_img = hsi_tensor
        current_mask = None
        if current_hsi_img is not None:
            current_mask = (current_hsi_img != 0).any(dim=0, keepdim=True).float()

        if self.resize_enable:
            if current_rgb_img is not None and self.rgb_img_size:
                current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'resize', self.rgb_img_size, is_label=False)
            if current_hsi_img is not None and self.hsi_img_size:
                current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'resize', self.hsi_img_size, is_label=False)
            if current_mask is not None and self.hsi_img_size:
                current_mask = self._apply_single_geometric_transform(current_mask, 'resize', self.hsi_img_size, is_label=True)
            if current_label_pil is not None and self.label_img_size:
                current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'resize', self.label_img_size, is_label=True)

        can_do_random = self.is_train

        if can_do_random and self.rotate_enable and self.rotate_degrees:
            angle = random.uniform(self.rotate_degrees[0], self.rotate_degrees[1])
            current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'rotate', angle, is_label=False)
            current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'rotate', angle, is_label=False)
            current_mask = self._apply_single_geometric_transform(current_mask, 'rotate', angle, is_label=True)
            current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'rotate', angle, is_label=True)

        if can_do_random and self.scale_enable and self.scale_limit:
            scale_factor = random.uniform(self.scale_limit[0], self.scale_limit[1])
            current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'scale', scale_factor, is_label=False)
            current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'scale', scale_factor, is_label=False)
            current_mask = self._apply_single_geometric_transform(current_mask, 'scale', scale_factor, is_label=True)
            current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'scale', scale_factor, is_label=True)

        if can_do_random and self.crop_enable and self.crop_size_cfg_tuple:
            img_for_params = current_rgb_img if current_rgb_img is not None else (current_hsi_img if current_hsi_img is not None else current_label_pil)
            if img_for_params is not None:
                cur_h, cur_w = self._get_image_dims(img_for_params)
                tgt_h, tgt_w = self.crop_size_cfg_tuple
                can_crop = (cur_h >= tgt_h and cur_w >= tgt_w)
                if can_crop and current_rgb_img is not None:
                    h, w = self._get_image_dims(current_rgb_img)
                    if not (h >= tgt_h and w >= tgt_w):
                        can_crop = False
                if can_crop and current_hsi_img is not None:
                    h, w = self._get_image_dims(current_hsi_img)
                    if not (h >= tgt_h and w >= tgt_w):
                        can_crop = False
                if can_crop and current_label_pil is not None:
                    h, w = self._get_image_dims(current_label_pil)
                    if not (h >= tgt_h and w >= tgt_w):
                        can_crop = False
                if can_crop:
                    crop_params = T.RandomCrop.get_params(img_for_params, output_size=self.crop_size_cfg_tuple)
                    current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'crop', crop_params, is_label=False)
                    current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'crop', crop_params, is_label=False)
                    current_mask = self._apply_single_geometric_transform(current_mask, 'crop', crop_params, is_label=True)
                    current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'crop', crop_params, is_label=True)

        final_rgb_size = self.crop_target_size_rgb if (self.crop_enable and self.crop_target_size_rgb) else (self.rgb_img_size if self.resize_enable else None)
        final_hsi_size = self.crop_target_size_hsi if (self.crop_enable and self.crop_target_size_hsi) else (self.hsi_img_size if self.resize_enable else None)
        final_lbl_size = self.crop_target_size_label if (self.crop_enable and self.crop_target_size_label) else (self.label_img_size if self.resize_enable else None)

        if current_rgb_img is not None and final_rgb_size:
            current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'resize', final_rgb_size, is_label=False)
        if current_hsi_img is not None and final_hsi_size:
            current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'resize', final_hsi_size, is_label=False)
        if current_mask is not None and final_hsi_size:
            current_mask = self._apply_single_geometric_transform(current_mask, 'resize', final_hsi_size, is_label=True)
        if current_label_pil is not None and final_lbl_size:
            current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'resize', final_lbl_size, is_label=True)

        if can_do_random and self.flip_enable:
            do_h = random.random() < self.p_horizontal
            do_v = random.random() < self.p_vertical
            current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'hflip', do_h, is_label=False)
            current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'hflip', do_h, is_label=False)
            current_mask = self._apply_single_geometric_transform(current_mask, 'hflip', do_h, is_label=True)
            current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'hflip', do_h, is_label=True)

            current_rgb_img = self._apply_single_geometric_transform(current_rgb_img, 'vflip', do_v, is_label=False)
            current_hsi_img = self._apply_single_geometric_transform(current_hsi_img, 'vflip', do_v, is_label=False)
            current_mask = self._apply_single_geometric_transform(current_mask, 'vflip', do_v, is_label=True)
            current_label_pil = self._apply_single_geometric_transform(current_label_pil, 'vflip', do_v, is_label=True)

        if current_hsi_img is not None and current_mask is not None:
            mask_bin = (current_mask > 0.5).to(current_hsi_img.dtype)
            current_hsi_img = current_hsi_img * mask_bin

        if self.is_train and self.rgb_cj_enable and self.rgb_cj is not None and current_rgb_img is not None:
            current_rgb_img = self.rgb_cj(current_rgb_img)

        if current_hsi_img is not None and self.hsi_norm_enable:
            current_hsi_img = self._apply_normalization(current_hsi_img, 'hsi')

        if current_hsi_img is not None and self.hsi_spectral_pad_enable and self.hsi_target_spectral_channels is not None:
            current_hsi_img = self._pad_hsi_spectral(current_hsi_img)

        if current_rgb_img is not None:
            current_rgb_img = self._apply_normalization(current_rgb_img, 'rgb')

        return current_rgb_img, current_hsi_img, current_label_pil

    def __call__(self, sample):
        input_image_data = sample.get('image')
        label_input_pil = sample.get('label')

        rgb_in = None
        hsi_in = None

        if self.data_modality_to_load == 'rgb_hsi':
            if isinstance(input_image_data, torch.Tensor):
                if input_image_data.shape[0] > 3:
                    rgb_in = input_image_data[:3, :, :]
                    hsi_in = input_image_data[3:, :, :]
                else:
                    raise ValueError(f"In rgb_hsi mode, input tensor has {input_image_data.shape[0]} channels; cannot split into RGB+HSI.")
            elif isinstance(input_image_data, dict):
                rgb_in = input_image_data.get('rgb')
                hsi_in = input_image_data.get('hsi')
            else:
                raise TypeError(f"Expected dict or tensor for 'image' in rgb_hsi mode, got {type(input_image_data)}")
        elif self.data_modality_to_load == 'rgb':
            rgb_in = input_image_data
        elif self.data_modality_to_load == 'hsi':
            hsi_in = input_image_data

        if not isinstance(label_input_pil, Image.Image) and label_input_pil is not None:
            raise TypeError(f"Expected PIL for label, got {type(label_input_pil)}")

        rgb_out, hsi_out, label_out_pil = self._apply_geometric_transforms_sequentially(rgb_in, hsi_in, label_input_pil)

        final_label_tensor = None
        if label_out_pil is not None:
            if not isinstance(label_out_pil, Image.Image):
                raise TypeError(f"Label must be PIL after transforms, got {type(label_out_pil)}")
            label_np = np.array(label_out_pil, dtype=np.int64)
            if label_np.ndim == 3 and label_np.shape[-1] == 1:
                label_np = label_np.squeeze(axis=-1)
            final_label_tensor = torch.from_numpy(label_np).long()

        out = {}
        if self.data_modality_to_load == 'rgb_hsi':
            if rgb_out is not None:
                out['rgb'] = rgb_out
            if hsi_out is not None:
                out['hsi'] = hsi_out
        elif self.data_modality_to_load == 'rgb':
            if rgb_out is not None:
                out['image'] = rgb_out
        elif self.data_modality_to_load == 'hsi':
            if hsi_out is not None:
                out['image'] = hsi_out

        if final_label_tensor is not None:
            out['label'] = final_label_tensor
        elif label_input_pil is not None:
            print("Warning (Augmentation): Label provided but conversion failed.")

        return out

    def _apply_spectral_jitter(self, hsi_tensor):
        if not self.hsi_sj_enable or hsi_tensor is None:
            return hsi_tensor

        valid_mask = (hsi_tensor != 0).any(dim=0)
        if not valid_mask.any():
            return hsi_tensor

        C, H, W = hsi_tensor.shape
        pixels = hsi_tensor.permute(1, 2, 0)
        valid_pixels = pixels[valid_mask]

        if self.hsi_sj_contrast > 0:
            factors = torch.empty(valid_pixels.shape[0], 1, device=hsi_tensor.device).uniform_(1 - self.hsi_sj_contrast, 1 + self.hsi_sj_contrast)
            valid_pixels *= factors

        if self.hsi_sj_brightness > 0:
            add = torch.empty(valid_pixels.shape[0], 1, device=hsi_tensor.device).uniform_(-self.hsi_sj_brightness, self.hsi_sj_brightness)
            valid_pixels += add

        pixels[valid_mask] = valid_pixels
        return pixels.permute(2, 0, 1)