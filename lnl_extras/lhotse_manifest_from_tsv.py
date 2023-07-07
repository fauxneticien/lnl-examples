import glob
import os

import pandas as pd

from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map

from lhotse import (
    load_manifest,
    CutSet,
    RecordingSet,
    SupervisionSet,
    SupervisionSegment
)

# Function to process a single tsv, define here in order to make pickle-able
# use tqdm.contrib.concurrent.process_map for parallel processing
def lhotse_manifest_from_tsv(part_tsv, all_clips = None, clips_subdir='clips'):

    # If function was not called from lhotse_manifest_from_tsvs where all_clips is already computed
    if all_clips is None:
        clips_dir = part_tsv.parent / clips_subdir
        all_clips = RecordingSet.from_dir(clips_dir, pattern='*.wav', num_jobs=os.cpu_count() - 1) 

    part_name = part_tsv.name.replace('.tsv', '')

    part_sups_df = pd.read_csv(part_tsv, sep="\t")
    part_sups_df['lhotse_id'] = part_sups_df.path.apply(lambda x: os.path.basename(x).replace('.wav', ''))

    part_sups = SupervisionSet.from_segments([
        SupervisionSegment(
            id=f"{part_name}_{sup_id}",
            recording_id=rec_id,
            start=0,
            duration=all_clips[rec_id].duration,
            text=text
        )
        for (sup_id, (rec_id, text))
        in part_sups_df[['lhotse_id', 'text']].iterrows()
    ])

    part_recs = all_clips.filter(lambda c: c.id in part_sups_df.lhotse_id.to_list())

    return {
        'recordings': part_recs,
        'supervisions': part_sups
    }

# Function to process all tsvs in a directory
def lhotse_manifest_from_tsvs(data_dir, clips_subdir='clips'):

    data_dir = Path(data_dir)
    clips_dir = data_dir / clips_subdir

    all_clips = RecordingSet.from_dir(clips_dir, pattern='*.wav', num_jobs=os.cpu_count() - 1)

    part_tsvs  = list(data_dir.glob('*.tsv'))
    part_names = [ p.name.replace('.tsv', '') for p in part_tsvs ]

    func_wrapper = partial(lhotse_manifest_from_tsv, all_clips=all_clips)

    manifests_list = process_map(func_wrapper, part_tsvs)
    manifests_dict = dict(zip(part_names, manifests_list))

    return manifests_dict
