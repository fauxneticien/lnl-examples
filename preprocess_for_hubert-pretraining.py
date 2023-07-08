import argparse
import json
import warnings

from tqdm import tqdm
from pathlib import Path

from lhotse import CutSet, Fbank
from lhotse.shar import ArrayTarWriter
from lnl_extras import lhotse_manifest_from_tsvs

# Ignore warning triggered by Lhotse still using old PyTorch Audio API
warnings.filterwarnings("ignore", message="File-like object support in sox_io backend is deprecated")

parser = argparse.ArgumentParser(
    prog='preprocess_for_hubert-pretrain.py',
    description='Acoustic Unit Discovery and Pseudo-label creation for HuBERT pre-training'
)

parser.add_argument('dataset_name')
parser.add_argument('-dp', '--data_path', default='data')
parser.add_argument('-sp', '--shar_path', default='_shar')
parser.add_argument('-sz', '--shard_size', default=1000, type=int)

args = parser.parse_args()

data_path = Path(args.data_path)
shar_path = data_path / args.shar_path

dataset_path = data_path / args.dataset_name

dataset_manifests = lhotse_manifest_from_tsvs(dataset_path)
dataset_manifests_shar = { key: None for key in dataset_manifests.keys() }

for part_name, manifests_dict in dataset_manifests.items():

    output_dir = shar_path / args.dataset_name / part_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cuts = CutSet.from_manifests(**manifests_dict)

    print(f"Writing shar files for '{part_name}' ...")
    shar_files_paths = cuts.to_shar(output_dir, fields={"recording": "wav"}, shard_size=args.shard_size)

    cuts_shar = CutSet.from_shar(shar_files_paths)

    fbank = Fbank()

    print(f"Writing feature files for '{part_name}' ...")
    with ArrayTarWriter(
            f"{output_dir}/fbank.%06d.tar", shard_size=args.shard_size, compression="lilcom"
        ) as writer:
            
            for cut in tqdm(cuts_shar):

                # `feats` is a numpy array with log Mel filter bank features.
                feats = cut.compute_features(fbank)

                # `cut` now contains a field `cut.fbank` with metadata manifest for the features,
                # and a method `cut.load_fbank()` that loads the features (respects pad/truncation).
                cut = cut.attach_tensor(
                    "fbank", feats, frame_shift=fbank.frame_shift, temporal_dim=0
                )

                # We store the features under key `cut.id`, because during loading we'll check that the IDs match
                # to avoid data errors. We also store the feature manifest to have some information about this data.
                writer.write(cut.id, feats, cut.fbank)

    shar_files_paths['feature'] = writer.output_paths

    dataset_manifests_shar[part_name] = shar_files_paths

with open(shar_path / args.dataset_name / 'shar-paths.json', 'w') as fp:
    json.dump(dataset_manifests_shar, fp, indent=2)
