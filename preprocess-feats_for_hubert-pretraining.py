import argparse
import json
import torch
import warnings

import numpy as np

from tqdm import tqdm
from pathlib import Path

from lhotse import CutSet, Fbank
from lhotse.shar import ArrayTarWriter
from lnl_extras import lhotse_manifest_from_tsvs
from sklearn.cluster import MiniBatchKMeans
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import AutoModel

# Ignore warning triggered by Lhotse still using old PyTorch Audio API
warnings.filterwarnings("ignore", message="File-like object support in sox_io backend is deprecated")

parser = argparse.ArgumentParser(
    prog='preprocess_for_hubert-pretrain.py',
    description='Acoustic Unit Discovery and Pseudo-label creation for HuBERT pre-training'
)

# Positional arguments
parser.add_argument('dataset_name')
parser.add_argument('hf_model')
parser.add_argument('layer', type=int)

# Optional arguments
parser.add_argument('-dp', '--data_path', default='data')
parser.add_argument('-sp', '--shar_path', default='_shar')
parser.add_argument('-sz', '--shard_size', default=1000, type=int)

args = parser.parse_args()

hf_model = AutoModel.from_pretrained(args.hf_model)

assert hf_model.__class__.__name__ in {"Wav2Vec2Model", "HubertModel"}
ta_model = import_huggingface_model(hf_model).eval().to('cuda')

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
    
    with ArrayTarWriter(f"{output_dir}/fbank.%06d.tar", shard_size=args.shard_size, compression="lilcom") as fbank_writer, \
         ArrayTarWriter(f"{output_dir}/w2v2.%06d.tar", shard_size=args.shard_size, compression="lilcom") as w2v2_writer:
            
            for cut in tqdm(cuts_shar):

                audio_samples = torch.tensor(cut.load_audio())

                feats_fbank = cut.compute_features(fbank)

                fbank_frames, fbank_dim = feats_fbank.shape

                # Make number of frames divisible by two
                if fbank_frames % 2 != 0:
                     fbank_pad = np.zeros([1, fbank_dim])
                     feats_fbank = np.concatenate([feats_fbank, fbank_pad], axis=0)

                assert fbank.frame_shift == 0.01, "The aggregation code below assumes fbank frame shift is 0.01s"

                # Aggregate fbank frames to have same length as w2v2/hubert
                feats_fbank = feats_fbank.reshape(-1, 2, fbank_dim).mean(axis=1)

                with torch.inference_mode():
                    # Extract encoder features up to layer X
                    hidden_outputs, _ = ta_model.extract_features(
                         audio_samples.to('cuda'),
                         num_layers=args.layer
                    )

                # Fetch only last layer's (i.e. layer X) and convert to numpy array to store in Lhotse shar
                feats_w2v2 = hidden_outputs[args.layer - 1].squeeze(0).cpu().numpy()

                fbank_frames, fbank_dim = feats_fbank.shape
                w2v2_frames, w2v2_dmin = feats_w2v2.shape

                assert abs(fbank_frames - w2v2_frames) <= 1, "Difference in frames between fbank and w2v2 features greater than 1!"

                # Pad the shorter one
                if fbank_frames > w2v2_frames:
                     w2v2_pad = np.zeros([1, w2v2_dmin])
                     feats_w2v2 = np.concatenate([feats_w2v2, w2v2_pad], axis=0)

                elif w2v2_frames > fbank_frames:
                     feats_fbank = np.concatenate([feats_fbank, fbank_pad], axis=0)

                # They better be equal now!
                assert feats_w2v2.shape[0] == feats_fbank.shape[0]

                cut = cut.attach_tensor("fbank", feats_fbank, frame_shift=0.02, temporal_dim=0)
                cut = cut.attach_tensor("w2v2", feats_w2v2, frame_shift=0.02, temporal_dim=0)

                fbank_writer.write(cut.id, feats_fbank, cut.fbank)
                w2v2_writer.write(cut.id, feats_w2v2, cut.w2v2)
                    
    shar_files_paths['fbank'] = fbank_writer.output_paths
    shar_files_paths['w2v2']  = w2v2_writer.output_paths

    dataset_manifests_shar[part_name] = shar_files_paths
