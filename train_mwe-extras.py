import torch
import torchaudio
import lightning.pytorch as pl
import torch.nn.functional as F

from lightning import seed_everything
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import BucketingSampler, OnTheFlyFeatures
from lhotse.dataset.collation import TokenCollater
from lhotse.recipes import download_librispeech, prepare_librispeech
from torch.utils.data import DataLoader

seed_everything(42)

MAX_STEPS=10_000
LEARNING_RATE=1e-4

download_librispeech(dataset_parts="mini_librispeech")
libri = prepare_librispeech(corpus_dir="LibriSpeech", output_dir="data/")

cuts_train = CutSet.from_manifests(**libri["train-clean-5"])

tokenizer = TokenCollater(cuts_train)

class MinimalASRDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        self.tokenizer = tokenizer

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()
        feats, feat_lens = self.extractor(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        return {"inputs_padded": feats, "input_lengths": feat_lens, "labels_padded": tokens, "label_lengths": token_lens}

train_sampler = BucketingSampler(cuts_train, max_duration=300, shuffle=True, drop_last=True)

train_loader = DataLoader(
    MinimalASRDataset(),
    sampler=train_sampler,
    batch_size=None,
    num_workers=8
)

class DeepSpeechLightningModule(pl.LightningModule):
    def __init__(self, n_feature, tokenizer):
        super().__init__()
        # See Model section for details
        self.model    = torchaudio.models.DeepSpeech(n_feature=n_feature, n_class=len(list(tokenizer.idx2token)))
        # See Loss section for details
        self.ctc_loss = torch.nn.CTCLoss(blank=list(tokenizer.idx2token).index('<pad>'), reduction="sum", zero_infinity=True)

    def training_step(self, batch, batch_idx):
        # See loss usage section for details
        log_probs   = self.model(batch['inputs_padded'])
        
        loss = self.ctc_loss(log_probs.transpose(0, 1), batch["labels_padded"], batch["input_lengths"], batch["label_lengths"])

        # For multi-GPU training, normalize the loss based on the sum of batch_size across all GPUs
        batch_size = batch['inputs_padded'].size(0)
        # Get batch sizes from all GPUs
        batch_sizes = self.all_gather(batch_size)
        # Normalize by world size / batch size
        loss *= batch_sizes.size(0) / batch_sizes.sum()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=MAX_STEPS, anneal_strategy='linear')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1 }
        }

trainer = pl.Trainer(
    max_steps=MAX_STEPS,
    max_epochs=-1,
    strategy="ddp",
    use_distributed_sampler=False,
    # Disabled for this minimal working example
    enable_checkpointing=False,
    enable_model_summary=False,
    logger=None,
    limit_val_batches=0
)

trainer.fit(
  DeepSpeechLightningModule(80, tokenizer),
  train_dataloaders=train_loader
)
