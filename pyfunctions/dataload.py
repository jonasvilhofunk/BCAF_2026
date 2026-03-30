import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import traceback
from .augmentations import Augmentation


def _load_stats_from_file(stats_path):
    with open(stats_path, 'r') as f:
        return yaml.safe_load(f)


def _find_label_path(stem, label_base_path, candidates=None):
    if not label_base_path or not label_base_path.exists():
        return None
    if candidates is None:
        candidates = [f"{stem}.png"]
    for candidate in candidates:
        path = label_base_path / candidate
        if path.exists():
            return path
    return None


class SpectralWasteDataset(Dataset):
    DEFAULT_STATS_PATH = Path("/home/jon86439/BCAF_2026/datasets/spectralwaste_npz/hsi_stats.yaml")
    IMAGE_ROOT = Path("/home/jon86439/BCAF_2026/datasets/spectralwaste_npz")
    LABELS_ROOT = Path("/home/jon86439/spectralwaste_segmentation")
    NUM_CLASSES = 7
    CLASS_NAMES = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]

    def __init__(self, split, modality, transform):
        self.split = split
        self.modality = modality.lower()
        self.transform = transform

        self.data_root = self.IMAGE_ROOT
        self.images_dir = self.data_root / 'images' / split
        self.hsi_labels_dir = self.LABELS_ROOT / 'labels_hyper_lt' / split
        self.rgb_labels_dir = self.LABELS_ROOT / 'labels_rgb' / split

        self.image_file_sets = []
        self.image_paths = []
        self.label_paths = []

        self._load_files()

    def _load_files(self):
        npz_files = sorted(self.images_dir.glob('*.npz'))
        print(f"SpectralWasteDataset ({self.modality}, {self.split}): {len(npz_files)} files")

        if not npz_files:
            print(f"Warning: No data found at {self.images_dir}")
            return

        for npz_path in npz_files:
            stem = npz_path.stem
            label_path = self._get_label_path(stem)

            if self.modality == 'rgb_hsi':
                self.image_file_sets.append({'image_npy_path': npz_path, 'label_path': label_path})
            else:
                self.image_paths.append(npz_path)
                self.label_paths.append(label_path)

    def _get_label_path(self, stem):
        if self.modality == 'hsi':
            return _find_label_path(stem, self.hsi_labels_dir, [f"{stem}.png"])
        elif self.modality in ('rgb', 'rgb_hsi'):
            return _find_label_path(stem, self.rgb_labels_dir, [f"{stem}_grayscale.png", f"{stem}.png"])
        return None

    def __len__(self):
        return len(self.image_file_sets) if self.modality == 'rgb_hsi' else len(self.image_paths)

    def __getitem__(self, idx):
        try:
            if self.modality == 'rgb_hsi':
                image_for_aug = self._load_rgb_hsi(idx)
                label_pil = self._load_label_rgb_hsi(idx)
            else:
                image_for_aug = self._load_single_modality(idx)
                label_pil = self._load_label_single(idx)

            sample = {'image': image_for_aug, 'label': label_pil}
            if self.transform:
                sample = self.transform(sample)
            return sample

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            traceback.print_exc()
            return {}

    def _load_rgb_hsi(self, idx):
        npz_path = self.image_file_sets[idx]['image_npy_path']
        with np.load(npz_path) as npz_file:
            return {
                'rgb': torch.from_numpy(npz_file['rgb']).float(),
                'hsi': torch.from_numpy(npz_file['hsi']).float()
            }

    def _load_single_modality(self, idx):
        npz_path = self.image_paths[idx]
        with np.load(npz_path) as npz_file:
            key = 'rgb' if self.modality == 'rgb' else 'hsi'
            return torch.from_numpy(npz_file[key]).float()

    def _load_label_rgb_hsi(self, idx):
        label_path = self.image_file_sets[idx]['label_path']
        return Image.open(label_path)

    def _load_label_single(self, idx):
        return Image.open(self.label_paths[idx])


def create_datasets(config, dataset_name, modality):
    aug_config = config.get('augmentation', {})
    enabled = aug_config.get('enable', False)

    hsi_stats = None
    if dataset_name.lower() == 'spectralwaste' and modality in ('hsi', 'rgb_hsi'):
        try:
            hsi_stats = _load_stats_from_file(SpectralWasteDataset.DEFAULT_STATS_PATH)
            print(f"Loaded HSI stats from {SpectralWasteDataset.DEFAULT_STATS_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load HSI stats: {e}")

    if dataset_name.lower() != 'spectralwaste':
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if enabled:
        train_transform = Augmentation(config=aug_config, is_train=True, data_modality_to_load=modality, hsi_stats_dict=hsi_stats)
        val_transform = Augmentation(config=aug_config, is_train=False, data_modality_to_load=modality, hsi_stats_dict=hsi_stats)
    else:
        minimal_config = {k: aug_config.get(k, {'enable': False}) for k in ('rgb_normalization', 'hsi_normalization', 'pad_channels', 'pca_transform', 'resize')}
        minimal_transform = Augmentation(config=minimal_config, is_train=False, data_modality_to_load=modality, hsi_stats_dict=hsi_stats)
        train_transform = val_transform = minimal_transform

    train_ds = SpectralWasteDataset('train', modality, train_transform)
    val_ds = SpectralWasteDataset('val', modality, val_transform)
    test_ds = SpectralWasteDataset('test', modality, val_transform)

    print(f"Loaded {dataset_name} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={SpectralWasteDataset.NUM_CLASSES}")
    return train_ds, val_ds, test_ds, SpectralWasteDataset.NUM_CLASSES, SpectralWasteDataset.CLASS_NAMES


def create_dataloaders(train_dataset, val_dataset, config, batch_size_train, batch_size_val):
    num_workers = config.get('num_workers', 0)
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': config.get('pin_memory', True),
        'persistent_workers': num_workers > 0
    }
    if num_workers > 0:
        common_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=config.get('drop_last', False), **common_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, **common_kwargs)

    print(f"Dataloaders: batch_train={batch_size_train} batch_val={batch_size_val} workers={num_workers}")
    return train_loader, val_loader