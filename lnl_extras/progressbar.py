from lightning.pytorch.callbacks import ProgressBar

from tqdm import tqdm

class LhotseCompatibleProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True
        
        self.sanity_val_check_done = False
        self.sanity_val_check_steps = 0

    def disable(self):
        self.enable = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.sanity_val_check_done = True

    def on_train_start(self, trainer, pl_module):
        self.train_pbar = tqdm(total=trainer.max_steps)

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_pbar.set_description_str(f"Epoch: {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        if self.sanity_val_check_done:
            self.train_pbar.set_postfix_str(",".join([ f"{k}: {v:.2f}" for (k,v) in outputs.items() ]))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)  # don't forget this :)
        self.train_pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.train_pbar.close()

    def on_validation_start(self, trainer, pl_module):
        if not self.sanity_val_check_done:
            self.val_pbar = tqdm(desc="Running full epoch to estimate number of validation batches...")
        else:
            self.val_pbar = tqdm(desc=f"Running validation", total=self.sanity_val_check_steps)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)

        if not self.sanity_val_check_done:
            self.sanity_val_check_steps += 1
        else:
            self.val_pbar.update(1)

    def on_validation_end(self, trainer, pl_module):
        self.val_pbar.close()
