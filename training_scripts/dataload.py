import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import traceback
from .augmentations import Augmentation


def _load_stats_content_from_file(stats_file_path_str):
    """Load HSI stats YAML/JSON into a dict."""
    stats_path = Path(stats_file_path_str)
    with open(stats_path, 'r') as f:
        return yaml.safe_load(f)


class SpectralWasteDataset(Dataset):
    """Dataset for SpectralWaste in npz format supporting rgb, hsi, and rgb_hsi modalities."""
    DEFAULT_STATS_PATH = Path("/home/jon86439/BCAF/training_scripts/configs/SpectralWaste/data/hsi_stats.yaml")

    def __init__(self, split, modality, transform, labelled=True):
        """
        split: 'train' | 'val' | 'test'
        modality: 'rgb' | 'hsi' | 'rgb_hsi'
        transform: augmentation / preprocessing pipeline
        labelled: if False, loads from unlabelled tree and skips labels
        """
        base_root = Path("/home/jon86439/Experiment_MA_Publication/data/SpectralWaste")
        self.data_root = base_root / ("labelled_npz" if labelled else "unlabelled_1000_npz")
        self.stats_path = SpectralWasteDataset.DEFAULT_STATS_PATH
        self.num_classes = 7
        self.class_names = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]

        self.split = split
        self.modality = modality.lower()
        self.transform = transform
        self.labelled = labelled

        self.image_file_sets = []      # for rgb_hsi
        self.image_paths = []          # for rgb or hsi
        self.label_paths = [] if labelled else None

        unified_images_dir = self.data_root / 'images' / self.split
        hsi_label_base_path = (self.data_root / 'labels_hyper_lt' / self.split) if labelled else None
        rgb_label_base_path = (self.data_root / 'labels_rgb' / self.split) if labelled else None

        potential_npy_files = sorted(unified_images_dir.glob('*.npz'))
        print(f"SpectralWasteDataset ({self.modality}, {self.split}): {len(potential_npy_files)} files")

        if not potential_npy_files:
            print(f"Warning: No data found for split '{self.split}' at {unified_images_dir}")
            return

        for npz_path in potential_npy_files:
            stem = npz_path.stem
            label_path = None

            if self.modality == 'rgb_hsi':
                if labelled and rgb_label_base_path and rgb_label_base_path.exists():
                    for candidate in (f"{stem}_grayscale.png", f"{stem}.png"):
                        temp = rgb_label_base_path / candidate
                        if temp.exists():
                            label_path = temp
                            break
                self.image_file_sets.append({'image_npy_path': npz_path, 'label_path': label_path if labelled else None})

            elif self.modality == 'hsi':
                if labelled:
                    if not hsi_label_base_path or not hsi_label_base_path.exists():
                        print("Warning: HSI label directory missing.")
                        continue
                    label_path = hsi_label_base_path / f"{stem}.png"
                    if not label_path.exists():
                        print(f"Warning: Missing HSI label for {npz_path.name}")
                        continue
                self.image_paths.append(npz_path)
                if labelled:
                    self.label_paths.append(label_path)

            elif self.modality == 'rgb':
                if labelled:
                    if not rgb_label_base_path or not rgb_label_base_path.exists():
                        print("Warning: RGB label directory missing.")
                        continue
                    for candidate in (f"{stem}_grayscale.png", f"{stem}.png"):
                        temp = rgb_label_base_path / candidate
                        if temp.exists():
                            label_path = temp
                            break
                    if not label_path:
                        print(f"Warning: Missing RGB label for {npz_path.name}")
                        continue
                self.image_paths.append(npz_path)
                if labelled:
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_file_sets) if self.modality == 'rgb_hsi' else len(self.image_paths)

    def __getitem__(self, idx):
        try:
            label_pil = None

            if self.modality == 'rgb_hsi':
                file_set = self.image_file_sets[idx]
                with np.load(file_set['image_npy_path']) as npz_file:
                    rgb_np = npz_file['rgb']
                    hsi_np = npz_file['hsi']
                image_for_aug = {
                    'rgb': torch.from_numpy(rgb_np).float(),
                    'hsi': torch.from_numpy(hsi_np).float()
                }
                if self.labelled and file_set.get('label_path'):
                    label_pil = Image.open(file_set['label_path'])

            else:
                npz_path = self.image_paths[idx]
                with np.load(npz_path) as npz_file:
                    if self.modality == 'rgb':
                        image_for_aug = torch.from_numpy(npz_file['rgb']).float()
                    else:  # hsi
                        image_for_aug = torch.from_numpy(npz_file['hsi']).float()
                if self.labelled:
                    label_pil = Image.open(self.label_paths[idx])

            sample = {'image': image_for_aug, 'label': label_pil}
            if self.transform:
                sample = self.transform(sample)
            return sample

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            traceback.print_exc()
            return {}  # Fallback structure

def create_datasets(main_config, dataset_name, data_modality_to_load, load_labelled_data=True):
    """Create train/val/test datasets plus class metadata."""
    aug_config = main_config.get('augmentation', {})
    augmentations_enabled = aug_config.get('enable', False)

    hsi_stats_content = None
    if dataset_name.lower() == 'spectralwaste' and data_modality_to_load in ('hsi', 'rgb_hsi'):
        try:
            hsi_stats_content = _load_stats_content_from_file(SpectralWasteDataset.DEFAULT_STATS_PATH)
            print(f"Loaded HSI stats: {SpectralWasteDataset.DEFAULT_STATS_PATH}")
        except FileNotFoundError:
            print("Warning: HSI stats file missing; continuing without stats.")
        except Exception as e:
            print(f"Warning: Failed to load HSI stats: {e}")

    if augmentations_enabled:
        train_transform = Augmentation(
            config=aug_config,
            is_train=True,
            data_modality_to_load=data_modality_to_load,
            hsi_stats_dict=hsi_stats_content
        )
        val_transform = Augmentation(
            config=aug_config,
            is_train=False,
            data_modality_to_load=data_modality_to_load,
            hsi_stats_dict=hsi_stats_content
        )
    else:
        minimal_config = {
            'rgb_normalization': aug_config.get('rgb_normalization', {'enable': False}),
            'hsi_normalization': aug_config.get('hsi_normalization', {'enable': False}),
            'pad_channels': aug_config.get('pad_channels', {'enable': False}),
            'pca_transform': aug_config.get('pca_transform', {'enable': False}),
            'resize': aug_config.get('resize', {'enable': False})
        }
        minimal_transform = Augmentation(
            config=minimal_config,
            is_train=False,
            data_modality_to_load=data_modality_to_load,
            hsi_stats_dict=hsi_stats_content
        )
        train_transform = minimal_transform
        val_transform = minimal_transform

    if dataset_name.lower() != 'spectralwaste':
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    train_ds = SpectralWasteDataset('train', data_modality_to_load, train_transform, labelled=load_labelled_data)
    val_ds = SpectralWasteDataset('val', data_modality_to_load, val_transform, labelled=load_labelled_data)
    test_ds = SpectralWasteDataset('test', data_modality_to_load, val_transform, labelled=load_labelled_data)

    ref_ds = train_ds if len(train_ds) > 0 else (val_ds if len(val_ds) > 0 else test_ds)
    num_classes = ref_ds.num_classes
    class_names = ref_ds.class_names

    print(f"Loaded '{dataset_name}' | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")
    return train_ds, val_ds, test_ds, num_classes, class_names


def create_dataloaders(train_dataset, val_dataset, data_loading_config, training_batch_size, validation_batch_size):
    """Create DataLoader objects for training and validation."""
    num_workers = data_loading_config.get('num_workers', 0)
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': data_loading_config.get('pin_memory', True),
        'prefetch_factor': data_loading_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        'persistent_workers': True if num_workers > 0 else False
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        drop_last=data_loading_config.get('drop_last', False),
        **common_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        **common_kwargs
    )
    print(f"Dataloaders ready: batch_train={training_batch_size} batch_val={validation_batch_size} workers={num_workers}")
    return train_loader, val_loader
