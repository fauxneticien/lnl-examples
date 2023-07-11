import joblib
import torch

import numpy as np

from lhotse import CutSet
from lhotse.dataset import (
    DynamicBucketingSampler,
    IterableDatasetWrapper,
    make_worker_init_fn
)
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

train_path = Path('data/_shar/20230710_mini-librispeech/train/')

# Based on https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb#scrollTo=fGjbhx_0vLFv
train_cuts = CutSet.from_shar(
    fields={
        'cuts': sorted(list(train_path.glob("cuts.*.jsonl.gz"))),
        'w2v2': sorted(list(train_path.glob("w2v2.*.tar")))
    },
    # The three arguments below are specifically for dataloading.
    # shuffle_shards=True enables shuffling of shards,
    # stateful_shuffle=True makes the shuffling different on each epoch,
    # and seed="randomized" tells the CutSet to randomize the seed on each dataloader node and worker.
    shuffle_shards=True,
    stateful_shuffle=True,
    seed="randomized",
).repeat()  # repeat() enables the stateful shuffling

train_sampler = DynamicBucketingSampler(
    train_cuts,
    shuffle=True,
    # For k-means batch size of 10,000 frames
    # That's about 205 frames emitted at 49Hz (10_000/49 = 204.1)
    max_duration=250,
    # We'll drop the last batch which may have fewer than 10k frames
    drop_last=True,
    num_buckets=10,
    rank=0,
    world_size=1
)

class KMeansTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=10_000):
        self.batch_size = batch_size

    def __getitem__(self, cuts: CutSet) -> dict:
        w2v2_feats = np.concatenate([ c.load_w2v2() for c in cuts ], axis=0)
        return w2v2_feats[:self.batch_size, :]

train_iter_dataset = IterableDatasetWrapper(
    dataset=KMeansTrainingDataset(),
    sampler=train_sampler,
)

train_dloader = DataLoader(
    train_iter_dataset,
    batch_size=None,
    # For faster dataloading, use num_workers > 1
    num_workers=8,
    # Note: Lhotse offers its own "worker_init_fn" that helps properly
    #       set the random seeds in all workers (also with multi-node training)
    #       and randomizes the shard order across different workers.
    worker_init_fn=make_worker_init_fn(seed=0),
)

MAX_STEPS = 10_000

km_model = MiniBatchKMeans(
    n_clusters=500,
    init="k-means++",
    max_iter=MAX_STEPS,
    batch_size=10_000,
    verbose=0,
    compute_labels=False,
    tol=0.0,
    max_no_improvement=100,
    init_size=None,
    n_init=20,
    reassignment_ratio=0.0,
)

for (step, batch) in (pbar := tqdm(enumerate(train_dloader), total=MAX_STEPS)):

    if step >= MAX_STEPS:
        break

    km_model.partial_fit(batch)

joblib.dump(km_model, "tmp/km_model_20230710_mini-librispeech.joblib")
