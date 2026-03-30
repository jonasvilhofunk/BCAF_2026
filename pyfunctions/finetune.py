import os
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]  # .../BCAF
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Prefer local training_scripts, fallback to legacy scripts package
from training_scripts.build_model import build_model_finetune

from training_scripts.losses import SegmentationLoss, calculate_class_frequencies, calculate_class_weights_from_frequencies
from training_scripts.wandb_image_visualization import prepare_wandb_images_SpectralWaste
from training_scripts.metrics import calculate_metrics
from training_scripts.dataload import create_datasets, create_dataloaders
from training_scripts.warmup_scheduler import GradualWarmupScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SegmentationTrainer:
    """Trainer class for fine-tuning segmentation models"""
    def __init__(self, config):
        """Initialize the trainer with config"""
        self.config = config
        self.num_classes = None
        self.class_names = None

        # Set GPU device based on config
        if torch.cuda.is_available():
            gpu_ids_config = config['hardware']['gpu']
            if isinstance(gpu_ids_config, (int, str)):
                gpu_id = str(gpu_ids_config).split(',')[0]
            elif isinstance(gpu_ids_config, list) and gpu_ids_config:
                gpu_id = str(gpu_ids_config[0]).split(',')[0]
            else:
                print("Warning: GPU configuration invalid or empty. Defaulting to cuda:0 if available, else CPU.")
                gpu_id = '0'
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU: {gpu_id}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Create directories for saving
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(config['training']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        self.is_main_process = True  # Assuming single GPU
        self._setup_data()
        self.setup_model()
        self.setup_training()

        # Initialize wandb if enabled
        if self.config.get('wandb', {}).get('enable', False) and WANDB_AVAILABLE and self.is_main_process:
            wandb.init(
                project=self.config['wandb'].get('project', 'segmentation-finetuning'),
                name=self.config['wandb'].get('run_name', self.config.get('name', 'default-run')),
                config=self.config,
            )

    def _setup_data(self):
        """Handles dataset and dataloader creation by calling external functions."""
        self.modality = self.config['modality']
        load_labelled_data_flag = self.config.get('labelled', True)

        self.train_dataset, self.val_dataset, self.test_dataset, self.num_classes, self.class_names = create_datasets(
            main_config=self.config,
            dataset_name=self.config['dataset_name'],
            data_modality_to_load=self.modality,
            load_labelled_data=load_labelled_data_flag
        )

        print(f"Dataset '{self.config['dataset_name']}' loaded (labelled: {load_labelled_data_flag}): {self.num_classes} classes -> {self.class_names}")
        self.train_loader, self.val_loader = create_dataloaders(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            data_loading_config=self.config['data_loading'],
            training_batch_size=self.config['training']['batch_size'],
            validation_batch_size=self.config['training']['val_batch_size']
        )

        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['training'].get('eval_batch_size', self.config['training']['batch_size']),
                shuffle=False,
                num_workers=self.config['data_loading'].get('num_workers', 0),
                pin_memory=self.config['data_loading'].get('pin_memory', True),
                prefetch_factor=self.config['data_loading'].get('prefetch_factor', 2) if self.config['data_loading'].get('num_workers', 0) > 0 else None,
                persistent_workers=True if self.config['data_loading'].get('num_workers', 0) > 0 else False
            )
            print(f"Test loader created with {len(self.test_dataset)} samples.")
        else:
            self.test_loader = None
            print("No test dataset loaded.")

    def setup_model(self):
        """Initialize the segmentation model"""
        print(f"Setting up {self.config['modality']} segmentation model with {self.num_classes} classes.")
        self.model = build_model_finetune(self.config, num_classes=self.num_classes)
        self.model.to(self.device)

    def setup_training(self):
        training_config = self.config['training']
        label_key_for_freq_calc = 'label'

        final_class_weights = None
        weighting_config = training_config['loss'].get('weighting', {})
        weighting_strategy = weighting_config.get('strategy', 'none')

        if weighting_strategy in ['frequency', 'square_root'] and self.train_loader:
            class_frequencies = calculate_class_frequencies(
                self.train_loader,
                self.num_classes,
                label_key_for_freq_calc,
                self.device
            )
            if class_frequencies is not None:
                final_class_weights = calculate_class_weights_from_frequencies(
                    class_frequencies,
                    weighting_strategy,
                    self.device,
                    num_classes_for_norm=self.num_classes
                )
            else:
                print(f"Warning: Class frequency calculation for strategy '{weighting_strategy}' returned None. No weights applied.")
                weighting_strategy = 'none'
        elif weighting_strategy == 'custom' and 'custom_weights' in weighting_config:
            custom_weights_list = weighting_config['custom_weights']
            if custom_weights_list and len(custom_weights_list) == self.num_classes:
                final_class_weights = torch.tensor(custom_weights_list, dtype=torch.float, device=self.device)
            else:
                print(f"Warning: 'custom_weights' invalid or length mismatch. No weights applied.")
                weighting_strategy = 'none'
        else:
            weighting_strategy = 'none'
            final_class_weights = None

        loss_cfg_dict = training_config['loss']
        self.criterion = SegmentationLoss(
            loss_type=loss_cfg_dict.get('type', 'ce_dice'),
            num_classes=self.num_classes,
            ce_weight=loss_cfg_dict.get('ce_weight', 1.0),
            dice_weight=loss_cfg_dict.get('dice_weight', 1.0),
            focal_gamma=loss_cfg_dict.get('focal_gamma', 2.0),
            class_weights=final_class_weights
        ).to(self.device)

        if final_class_weights is not None:
            print(f"Loss function: {loss_cfg_dict.get('type', 'ce_dice')} with weighting: {weighting_strategy}. Applied weights: {final_class_weights.cpu().numpy().tolist()}")
        else:
            print(f"Loss function: {loss_cfg_dict.get('type', 'ce_dice')} with weighting: {weighting_strategy} (no weights applied).")

        # Optimizer setup using model's parameter groups
        self.setup_optimizer()

        # Scheduler setup
        total_epochs = training_config['epochs']
        warmup_epochs = training_config.get('warmup_epochs', 0)

        scheduler_type = training_config.get('lr_scheduler', {}).get('type', 'cosine')
        scheduler_params = training_config.get('lr_scheduler', {}).get('params', {})

        if scheduler_type == 'poly':
            from torch.optim.lr_scheduler import PolynomialLR
            self.scheduler = PolynomialLR(
                self.optimizer,
                total_iters=total_epochs - warmup_epochs if warmup_epochs > 0 else total_epochs,
                power=scheduler_params.get('power', 0.9)
            )
        else:
            self.scheduler = None

        if warmup_epochs > 0 and self.scheduler is not None:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1.0,
                total_epoch=warmup_epochs,
                after_scheduler=self.scheduler
            )

        # AMP
        self.use_amp = training_config.get('amp', True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        else:
            print("AMP disabled")

        self.best_val_miou = 0.0
        self.best_epoch = 0

    def setup_optimizer(self):
        """Setup optimizer using model's predefined parameter groups"""
        training_config = self.config['training']

        if hasattr(self.model, 'param_groups') and self.model.param_groups:
            print("Using model's predefined parameter groups for optimizer")
            print(f"Parameter group details for optimizer:")
            for i, group in enumerate(self.model.param_groups):
                param_count = sum(p.numel() for p in group['params'] if p.requires_grad)
                group_name = group.get('name', f'group_{i}')
                print(f"  - {group_name}: {param_count:,} trainable parameters, LR: {group['lr']:.2e}")

            self.optimizer = torch.optim.AdamW(
                self.model.param_groups,
                weight_decay=training_config.get('weight_decay', 0.01)
            )
        else:
            print("No predefined parameter groups found, using fallback setup")
            lr = training_config['learning_rate']
            head_params, backbone_params = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                (head_params if 'seg_head' in name else backbone_params).append(param)

            param_groups = []
            if backbone_params:
                backbone_lr = lr * training_config.get('backbone_lr_factor', 0.1)
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
                print(f"  - Backbone: {sum(p.numel() for p in backbone_params):,} params, LR: {backbone_lr:.2e}")

            if head_params:
                param_groups.append({'params': head_params, 'lr': lr})
                print(f"  - Head: {sum(p.numel() for p in head_params):,} params, LR: {lr:.2e}")

            if param_groups:
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=training_config.get('weight_decay', 0.01))
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=training_config.get('weight_decay', 0.01))
                print(f"  - All parameters: {sum(p.numel() for p in self.model.parameters()):,} params, LR: {lr:.2e}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        modality = self.config['modality']
        accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        for i, batch in pbar:
            if modality == 'rgb_hsi':
                inputs = {
                    'rgb': batch['rgb'].to(self.device, non_blocking=True),
                    'hsi': batch['hsi'].to(self.device, non_blocking=True)
                }
            else:
                inputs = batch['image'].to(self.device, non_blocking=True)

            labels = batch['label'].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                if outputs.shape[2:] != labels.shape[1:]:
                    outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
                loss = self.criterion(outputs, labels)

            if accum_steps > 1:
                loss = loss / accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                if self.config['training'].get('clip_grad_norm', 0) > 0 and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad_norm'])
                elif self.config['training'].get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad_norm'])

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps if accum_steps > 1 else loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / len(self.train_loader)

        if WANDB_AVAILABLE and self.config.get('wandb', {}).get('enable', False) and self.is_main_process:
            wandb.log({'train/epoch_loss': avg_loss, 'train/lr': self.optimizer.param_groups[0]['lr']}, step=epoch + 1)

        return avg_loss

    def validate_epoch(self, epoch, is_test=False):
        """Validate the model on the validation set or test set"""
        self.model.eval()
        total_loss = 0
        model_modality = self.config['modality']
        num_classes = self.num_classes

        current_loader = self.test_loader if is_test else self.val_loader
        if not current_loader:
            print(f"Warning: {'Test' if is_test else 'Validation'} loader not available. Skipping.")
            return (0, {}) if not is_test else {}

        desc_prefix = "Testing" if is_test else "Validating"
        log_prefix = "test" if is_test else "val"

        all_preds_list, all_labels_list = [], []
        vis_images_list, vis_labels_list, vis_preds_list = [], [], []
        max_vis_samples = self.config.get('visualization', {}).get('max_images', 4) if not is_test else 0

        # control visualization frequency via visualization.log_interval (default every epoch)
        vis_cfg = self.config.get('visualization', {})
        vis_max_images = vis_cfg.get('max_images', 4)
        vis_log_interval = int(vis_cfg.get('log_interval', 1))
        if is_test:
            max_vis_samples = 0
        else:
            max_vis_samples = vis_max_images if ((epoch + 1) % max(1, vis_log_interval) == 0) else 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(current_loader, desc=f"{desc_prefix} Epoch {epoch+1}", leave=False)):
                labels = batch['label'].to(self.device, non_blocking=True)

                if model_modality == 'rgb_hsi':
                    images = {
                        'rgb': batch['rgb'].to(self.device, non_blocking=True),
                        'hsi': batch['hsi'].to(self.device, non_blocking=True)
                    }
                elif model_modality in ('rgb', 'hsi'):
                    images = batch['image'].to(self.device, non_blocking=True)
                else:
                    raise ValueError(f"Unsupported model_modality for validation: {model_modality}")

                with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                    logits = self.model(images)
                    if logits.shape[2:] != labels.shape[1:]:
                        logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
                    loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds_list.append(preds.cpu())
                all_labels_list.append(labels.cpu())

                # Visualization data collection
                if not is_test and batch_idx < (max_vis_samples // current_loader.batch_size) + 1:
                    if isinstance(images, dict):  # rgb_hsi
                        img_to_vis = images.get('rgb', images.get('hsi'))
                        if img_to_vis is not None:
                            vis_images_list.append(img_to_vis.cpu())
                    else:
                        vis_images_list.append(images.cpu())
                    vis_labels_list.append(labels.cpu())
                    vis_preds_list.append(logits.cpu())

        avg_loss = total_loss / len(current_loader)
        all_preds = torch.cat(all_preds_list).numpy()
        all_labels = torch.cat(all_labels_list).numpy()
        metrics = calculate_metrics(all_preds, all_labels, num_classes)

        # Prepare only mIoU + per-class IoU for WandB (validation only), and include sample images+predictions
        wandb_imgs_log = {}
        if not is_test and self.is_main_process and WANDB_AVAILABLE and self.config.get('wandb', {}).get('enable', False):
            # Concatenate full epoch tensors for IoU calculation
            labels_cat = torch.cat(all_labels_list)
            preds_cat = torch.cat(all_preds_list)

            # Prepare images for visualization (use collected vis images if available, otherwise a minimal dummy)
            if vis_images_list:
                vis_images_cat = torch.cat(vis_images_list)[:max_vis_samples]
            else:
                # create a dummy RGB image tensor shaped (N,3,H,W) using label spatial dims
                H, W = labels_cat.shape[1], labels_cat.shape[2]
                n = min(labels_cat.shape[0], max_vis_samples)
                vis_images_cat = torch.zeros((n, 3, H, W), dtype=torch.float)

            class_names = self.class_names if self.class_names is not None else [f"Class {i}" for i in range(self.num_classes)]
            modality_for_vis = 'rgb' if model_modality == 'rgb_hsi' else model_modality

            # Prefer the helper that returns both IoU metrics and images
            try:
                from training_scripts.wandb_image_visualization import prepare_wandb_logs_SpectralWaste
            except Exception:
                prepare_wandb_logs_SpectralWaste = None

            if prepare_wandb_logs_SpectralWaste is not None:
                iou_metrics, images_list = prepare_wandb_logs_SpectralWaste(
                    vis_images_cat, labels_cat, preds_cat,
                    num_classes=self.num_classes,
                    modality=modality_for_vis,
                    class_names=class_names,
                    max_images=max_vis_samples,
                    split='val'
                )
                # Map returned keys to WandB-friendly keys (val/... )
                log_dict = {}
                if isinstance(iou_metrics, dict):
                    # mIoU
                    miou_val = iou_metrics.get('val_mIoU', iou_metrics.get('mIoU', None))
                    if miou_val is not None:
                        log_dict[f'{log_prefix}/miou'] = miou_val
                    # per-class IoU keys are expected as 'val_iou_<class_name>'
                    for i in range(self.num_classes):
                        cname = class_names[i] if i < len(class_names) else f"class_{i}"
                        key = f'val_iou_{cname}'
                        if key in iou_metrics:
                            log_dict[f'{log_prefix}/iou/{cname}'] = iou_metrics[key]

                    if images_list:
                        wandb_imgs_log = {f'{log_prefix}/samples': images_list}

                    if log_dict:
                        wandb.log({**log_dict, **wandb_imgs_log}, step=epoch + 1)
                        # avoid logging other metrics (precision/recall/f1/support) as requested
            else:
                # Fallback: if helper not available, only log overall miou + per-class IoU from calculate_metrics if present
                log_dict = {}
                if 'miou' in metrics:
                    log_dict[f'{log_prefix}/miou'] = metrics['miou']
                if 'class_iou' in metrics:
                    for i in range(num_classes):
                        cname = class_names[i] if i < len(class_names) else f"Class_{i}"
                        log_dict[f'{log_prefix}/iou/{cname}'] = metrics['class_iou'][i]
                if log_dict:
                    wandb.log(log_dict, step=epoch + 1)

        print(f"{desc_prefix} Epoch {epoch+1} - Loss: {avg_loss:.4f}, mIoU: {metrics['miou']:.4f}, Acc: {metrics['accuracy']:.4f}, F1-macro: {metrics.get('f1', metrics.get('f1_macro', 0.0)):.4f}")

        if is_test:
            return metrics
        return avg_loss, metrics

    def save_checkpoint(self, epoch, is_best=False):
        """Save a checkpoint of the model, ensuring it's done only by the main process."""
        if not self.is_main_process:
            return

        checkpoint_name = "best.pth" if is_best else f"epoch_{epoch}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        def _uniformize_fusion_state_dict(sd):
            from collections import OrderedDict
            out = OrderedDict()
            # Only promote true backbone submodules; never touch seg_head or fusion_layer
            backbone_roots = ('backbone.', 'patch_embed.', 'layers.', 'norm.', 'pos_drop.', 'stages.', 'hf_model.')
            for k, v in sd.items():
                if k.startswith("rgb_model.") and not k.startswith("rgb_model.backbone."):
                    suffix = k[len("rgb_model."):]
                    if suffix.startswith(backbone_roots):
                        out[f"rgb_model.backbone.{suffix}"] = v
                    else:
                        out[k] = v  # keep seg_head and others as-is
                elif k.startswith("hsi_model.") and not k.startswith("hsi_model.backbone."):
                    suffix = k[len("hsi_model."):]
                    if suffix.startswith(backbone_roots):
                        out[f"hsi_model.backbone.{suffix}"] = v
                    else:
                        out[k] = v
                else:
                    out[k] = v
            return out

        model_state_dict = self.model.state_dict()
        # Apply normalization for fusion models (safe for others too)
        model_state_dict = _uniformize_fusion_state_dict(model_state_dict)

        checkpoint = {
            'epoch': epoch,
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.best_val_miou
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save final model (weights only)
        if is_best or epoch == self.config['training']['epochs']:
            model_name_prefix = self.config['training'].get('model_name', self.config.get('name', 'model'))
            final_model_name = f"{model_name_prefix}_best.pth" if is_best else f"{model_name_prefix}_epoch_{epoch}.pth"
            final_model_path = self.model_dir / final_model_name
            torch.save(model_state_dict, final_model_path)
            print(f"Model weights saved to {final_model_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to resume training"""
        resolved_path = Path(checkpoint_path)
        if not resolved_path.exists():
            resolved_path = self.checkpoint_dir / checkpoint_path
            if not resolved_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {resolved_path}")

        print(f"Loading checkpoint from {resolved_path}")
        checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)

        model_state_dict = checkpoint['model']
        if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
            print("Detected compiled model checkpoint, removing _orig_mod. prefixes...")
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                cleaned_state_dict[key[10:]] = value if key.startswith('_orig_mod.') else value
            model_state_dict = cleaned_state_dict

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded successfully from {resolved_path}")

        if checkpoint.get('scheduler') and self.scheduler:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}. Scheduler may restart.")

        self.best_val_miou = checkpoint.get('best_miou', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        self.best_epoch = start_epoch
        print(f"Resumed from epoch {start_epoch} with best mIoU {self.best_val_miou:.4f}")
        return start_epoch

    def train(self):
        """Main training loop"""
        start_epoch = 0
        for epoch in range(start_epoch, self.config['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            if self.scheduler:
                self.scheduler.step()

            val_loss, val_metrics = self.validate_epoch(epoch)
            current_miou = val_metrics['miou']

            if self.is_main_process:
                if current_miou > self.best_val_miou:
                    self.best_val_miou = current_miou
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(epoch + 1, is_best=True)

                save_interval = self.config['training'].get('save_checkpoint_interval', 0)
                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch + 1)

        if self.is_main_process:
            print("Training complete!")
            print(f"Best validation mIoU: {self.best_val_miou:.4f} (Epoch {self.best_epoch})")

            if self.test_loader is not None:
                best_checkpoint_path = self.checkpoint_dir / "best.pth"
                if best_checkpoint_path.exists():
                    print(f"\nLoading best model from {best_checkpoint_path} for testing...")
                    self.load_checkpoint(str(best_checkpoint_path))
                    test_metrics = self.validate_epoch(epoch=self.config['training']['epochs'] - 1, is_test=True)
                    print("\nTest Set Performance (using best model):")
                    for metric_name, metric_val in test_metrics.items():
                        if isinstance(metric_val, float):
                            print(f"  {metric_name}: {metric_val:.4f}")
                        if WANDB_AVAILABLE and self.config.get('wandb', {}).get('enable', False):
                            wandb.log({f"final_test/{metric_name}": metric_val})
                else:
                    print("Warning: best.pth not found, skipping final test evaluation.")
            else:
                print("No test loader configured, skipping final test evaluation.")

        if WANDB_AVAILABLE and self.config.get('wandb', {}).get('enable', False) and self.is_main_process:
            wandb.finish()


def main(args):
    """Main function"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config.get('training', {}).get('random_seed')
    if seed is not None:
        print(f"Setting random seed to: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        print("No random_seed specified in training config.")

    trainer = SegmentationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args)