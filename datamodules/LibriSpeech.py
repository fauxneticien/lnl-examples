
import lightning.pytorch as pl

from lhotse import CutSet
from lhotse.recipes import download_librispeech, prepare_librispeech
from lhotse.dataset.sampling import BucketingSampler

from torch.utils.data import DataLoader

from ._datasets import MinimalASRDataset
from lnl_extras import LnLTokenCollater

class LibriSpeechDataModule(pl.LightningDataModule):

    def __init__(self, lnl_cfg, dataset_parts, corpus_dir="./data/LibriSpeech"):
        super().__init__()
        self.prepare_data_per_node = False

        self.lnl_cfg = lnl_cfg
        self.dataset_parts = dataset_parts
        self.corpus_dir = corpus_dir

    def prepare_data(self,) -> None:
        download_librispeech(
            target_dir=self.corpus_dir,
            dataset_parts=self.dataset_parts.values()
        )

    def setup(self, stage = None):
        libri = prepare_librispeech(
            corpus_dir=f"{self.corpus_dir}/LibriSpeech",
            output_dir=f"{self.corpus_dir}/manifests/"
        )
        self.cuts_train = CutSet.from_manifests(**libri[self.dataset_parts.train])
        self.cuts_val = CutSet.from_manifests(**libri[self.dataset_parts.val])
        
        self.tokenizer = LnLTokenCollater(self.cuts_train)
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
            sampler=BucketingSampler(self.cuts_train, max_duration=self.lnl_cfg.bucket_max_dur, shuffle=True, drop_last=True),
            batch_size=None,
            num_workers=self.lnl_cfg.dloader_workers
        )
    
    def val_dataloader(self):

        return DataLoader(
            MinimalASRDataset(self.model_n_feature, self.tokenizer),
            sampler=BucketingSampler(self.cuts_val, max_duration=self.lnl_cfg.bucket_max_dur, shuffle=False, drop_last=True),
            batch_size=None,
            num_workers=self.lnl_cfg.dloader_workers
        )
