import joblib

from lhotse import CutSet
from lhotse.shar import ArrayTarWriter
from pathlib import Path
from tqdm import tqdm

dataset_name = '20230710_mini-librispeech'
km_model = joblib.load("tmp/km_model_20230710_mini-librispeech.joblib")

dataset_path = Path('data/_shar/') / dataset_name

for part_path in dataset_path.iterdir():

    part_cuts = CutSet.from_shar(
        fields={
            'cuts': sorted(list(part_path.glob("cuts.*.jsonl.gz"))),
            'w2v2': sorted(list(part_path.glob("w2v2.*.tar")))
        }
    )

    print(f"Generating pre-training labels for '{part_path.name}' ...")
    
    with ArrayTarWriter(f"{part_path}/ptlabel.%06d.tar", shard_size=1000, compression="numpy") as ptlabel_writer:

        for cut in tqdm(part_cuts):

            w2v2_feats = cut.load_w2v2()

            ptlabels = km_model.predict(w2v2_feats.astype(float))

            cut = cut.attach_tensor("ptlabel", ptlabels, frame_shift=0.02, temporal_dim=0)

            ptlabel_writer.write(cut.id, ptlabels, cut.ptlabel)
