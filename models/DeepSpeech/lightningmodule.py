import hydra
import jiwer
import torch
import torchinfo

import lightning.pytorch as pl
import pandas as pd

from .model import DeepSpeech

class DeepSpeechLightningModule(pl.LightningModule):

    def __init__(self, lnl_cfg, val_decoder, n_feature=80):
        super().__init__()
        # Set input feature dimension on init so DataModule can use info
        self.n_feature = n_feature

        self.lnl_cfg = lnl_cfg
        self.val_decoder = val_decoder

        self.val_losses = []
        self.val_ref_pred_pairs = []

    def setup(self, stage = None):
        self.model    = DeepSpeech(
            # Set in __init__() above
            n_feature=self.n_feature,
            # Set in LibrisDataModule setup()
            n_class=self.trainer.datamodule.tokenizer_n_class
        )

        output_labels = list(self.trainer.datamodule.tokenizer.idx2token)
        pad_token = output_labels.index('<pad>')

        # For details, see loss usage section in notebooks/01_mwe.ipynb
        self.ctc_loss = torch.nn.CTCLoss(blank=pad_token, reduction="sum", zero_infinity=True)

        self.val_decoder.setup(output_labels, pad_token)

        torchinfo.summary(self.model, (1, 100, self.n_feature))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lnl_cfg.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lnl_cfg.max_lr,
            pct_start=self.lnl_cfg.lr_warmup_pc,
            total_steps=self.lnl_cfg.max_updates,
            anneal_strategy='linear'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1 }
        }

    def _step(self, batch, batch_idx, step_type):
        # For details, see loss usage section in notebooks/01_mwe.ipynb
        log_probs   = self.model(batch['inputs_padded'])
        loss = self.ctc_loss(log_probs.transpose(0, 1), batch["labels_padded"], batch["input_lengths"], batch["label_lengths"])

        # For multi-GPU training, normalize the loss based on the sum of batch_size across all GPUs
        batch_size = batch['inputs_padded'].size(0)
        # Get batch sizes from all GPUs
        batch_sizes = self.all_gather(batch_size)
        # Normalize by world size / batch size
        loss *= batch_sizes.size(0) / batch_sizes.sum()

        self.log(f"{step_type}/loss", loss.item(), sync_dist=True, prog_bar=True)

        return loss, log_probs

    def training_step(self, batch, batch_idx):

        loss, _ = self._step(batch, batch_idx, "train")

        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, log_probs = self._step(batch, batch_idx, "val")

        refs  = self.trainer.datamodule.tokenizer.inverse(batch["labels_padded"], batch["label_lengths"])
        preds = self.val_decoder(log_probs)

        self.val_losses.append(loss.item())
        self.val_ref_pred_pairs += list(zip(refs, preds))

        return loss
    
    def on_validation_epoch_end(self, *_):
        val_loss_mean = torch.mean(torch.tensor(self.all_gather(self.val_losses)))

        refs_preds_df = pd.DataFrame(
            self.all_gather(self.val_ref_pred_pairs),
            columns=["Reference", "Prediction"]
        )

        # Free memory
        self.val_losses.clear()
        self.val_ref_pred_pairs.clear()

        # Only log the gathered metrices on main process
        if self.trainer.is_global_zero:
            wer = jiwer.wer(refs_preds_df['Reference'].to_list(), refs_preds_df['Prediction'].to_list())
            cer = jiwer.cer(refs_preds_df['Reference'].to_list(), refs_preds_df['Prediction'].to_list())

            self.log("val/loss", val_loss_mean, rank_zero_only=True, sync_dist=True)
            self.log("val/wer", wer, rank_zero_only=True, sync_dist=True)
            self.log("val/cer", cer, rank_zero_only=True, sync_dist=True)

            # Only print if training has already started (i.e. not after pre-training sanity check)
            if self.trainer.global_step > 0:
                print("------------------")
                print(refs_preds_df)
                print(f"{self.trainer.global_step=} {wer=:.2f}, {cer=:.2f}")
                print("------------------")
