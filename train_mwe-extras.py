import jiwer
import torch
import torchaudio

import lightning.pytorch as pl
import pandas as pd

from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import OnTheFlyFeatures
from lhotse.dataset.collation import TokenCollater
from lhotse.dataset.sampling import BucketingSampler
from lhotse.recipes import download_librispeech, prepare_librispeech

from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from torch.utils.data import DataLoader

class MinimalASRDataset(torch.utils.data.Dataset):
    def __init__(self, model_n_feature, tokenizer):
        self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=model_n_feature)))
        self.tokenizer = tokenizer

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()
        feats, feat_lens = self.extractor(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        return {"inputs_padded": feats, "input_lengths": feat_lens, "labels_padded": tokens, "label_lengths": token_lens}

class LibrisDataModule(pl.LightningDataModule):
    def prepare_data(self,) -> None:
        download_librispeech(dataset_parts="mini_librispeech")

    def setup(self, stage = None):
        libri = prepare_librispeech(corpus_dir="LibriSpeech", output_dir="data/")
        self.cuts_train = CutSet.from_manifests(**libri["train-clean-5"])
        self.cuts_val = CutSet.from_manifests(**libri["dev-clean-2"])
        
        self.tokenizer = TokenCollater(self.cuts_train)
        # Set number of total unique labels (i.e. output dimensions) to be fetched by model class during setup
        self.tokenizer_n_class = len(list(self.tokenizer.idx2token))

        # Fetch input dimensions model expects to pass to feature extractor
        # We fetch the attribute here instead of in train_dataloader(), after which
        # it no longer exists as self.trainer.model, since it will be wrapped in 
        # DistributedDataParallel() by Lightning
        self.model_n_feature = self.trainer.model.n_feature

    def train_dataloader(self):

        return DataLoader(
            MinimalASRDataset(self.model_n_feature, self.tokenizer),
            sampler=BucketingSampler(self.cuts_train, max_duration=300, shuffle=True, drop_last=True),
            batch_size=None,
            num_workers=8
        )
    
    def val_dataloader(self):

        return DataLoader(
            MinimalASRDataset(self.model_n_feature, self.tokenizer),
            sampler=BucketingSampler(self.cuts_val, max_duration=300, shuffle=False, drop_last=True),
            batch_size=None,
            num_workers=8
        )

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def setup(self, output_labels, pad_token=0):
        self.output_labels = output_labels
        self.pad_token = pad_token

    def forward(self, log_probs: torch.Tensor):
        indices = torch.argmax(log_probs, dim=-1)

        predictions = []

        for p in list(indices):
            unique_indices = torch.unique_consecutive(p, dim=-1)
            prediction = "".join([ self.output_labels[t] for t in unique_indices if t != self.pad_token ])
            predictions.append(prediction)

        return predictions

class DeepSpeechLightningModule(pl.LightningModule):

    def __init__(self, val_decoder, n_feature=80):
        super().__init__()
        # Set input feature dimension on init so DataModule can use info
        self.n_feature = n_feature

        self.val_decoder = val_decoder

        self.val_losses = []
        self.val_ref_pred_pairs = []

    def setup(self, stage = None):
        self.model    = torchaudio.models.DeepSpeech(
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LNL_CONFIG["lr"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LNL_CONFIG["lr"], total_steps=LNL_CONFIG["max_updates"], anneal_strategy='linear')

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

seed_everything(42)

LNL_CONFIG = {
    "lr": 1e-4,
    "max_updates": 10_000,
    "grad_acc": 1,
    "val_every_n_updates": 1000,
}

trainer = pl.Trainer(
    max_steps=LNL_CONFIG["max_updates"],
    accumulate_grad_batches=LNL_CONFIG["grad_acc"],
    max_epochs=-1,
    # Turns out val_check_interval is based on dataloader batch steps not update steps
    # Disable validation after every epoch then set validation to occurr after every N updates
    check_val_every_n_epoch=None,
    val_check_interval=LNL_CONFIG["val_every_n_updates"] * LNL_CONFIG["grad_acc"],
    # Use DDP and prevent Lightning from replacing Lhotse's DDP-compatible sampler
    strategy="ddp",
    use_distributed_sampler=False,
    callbacks=[ 
        LearningRateMonitor(logging_interval='step')
    ],
    # Disabled for this minimal working example
    enable_checkpointing=False,
    enable_model_summary=False,
    logger=CSVLogger("logs_lnl-examples", name="mwe-extras")
)

trainer.fit(
  DeepSpeechLightningModule(val_decoder=GreedyCTCDecoder()),
  LibrisDataModule()
)
