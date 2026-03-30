import torch
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr
    
    This scheduler first increases the learning rate linearly from the base value
    to a target value over a specified number of epochs (warmup phase), and then
    optionally switches to another scheduler for subsequent epochs.
    
    Args:
        optimizer: Optimizer being used for training
        multiplier: Target multiplier for the learning rate at the end of warmup. 
                    If multiplier = 1.0, the LR increases from 0 to base_lr.
                    If multiplier > 1.0, the LR increases from base_lr to base_lr * multiplier.
        total_epoch: Total number of epochs for warmup
        after_scheduler: Scheduler to use after warmup is complete (e.g., CosineAnnealingLR)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, last_epoch=-1):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer,last_epoch)
        
    def get_lr(self):
        if self.finished:
            if self.after_scheduler:
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        # Warmup phase
        if self.multiplier == 1.0: # Warmup from 0 to base_lr
            return [base_lr * (self.last_epoch+1)/self.total_epoch for base_lr in self.base_lrs]
        else: # Warmup from base_lr to base_lr * multiplier
            return [base_lr * ((self.multiplier-1)*(self.last_epoch+1)/self.total_epoch + 1) for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if self.finished:
            if self.after_scheduler:
                self.after_scheduler.step()  # Remove all parameters
            return

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch >= self.total_epoch:
            self.finished = True
            # set base lr for after_scheduler
            if self.multiplier != 1.0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.base_lrs[i] * self.multiplier
            if self.after_scheduler:
                self.after_scheduler.step()  # Remove the 0 parameter
        else:
            lrs = self.get_lr()
            for g, lr in zip(self.optimizer.param_groups, lrs):
                g['lr'] = lr
