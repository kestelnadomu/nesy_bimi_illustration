import torch

class EarlyStopping:
    def __init__(self, monitor="val_loss", patience=10, restore_best_weights=True, verbose=1):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best = float('inf')
        self.counter = 0
        self.best_state = None
        self.stopped_epoch = 0
        self.stop_training = False

    def step(self, current, model, epoch):
        if current < self.best:
            self.best = current
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", save_best_only=True, verbose=0):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf')

    def step(self, current, model):
        if not self.save_best_only or current < self.best:
            self.best = current
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"Model saved to {self.filepath}")


class ReduceLROnPlateau:
    def __init__(self, optimizer, monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = float(min_lr)
        self.verbose = verbose
        self.best = float('inf')
        self.counter = 0

    def step(self, current):
        if current < self.best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if old_lr > new_lr:
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f"ReduceLROnPlateau: Reducing learning rate to {new_lr}")
                self.counter = 0