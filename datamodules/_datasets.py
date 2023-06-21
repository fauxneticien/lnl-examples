import torch

from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import OnTheFlyFeatures

class MinimalASRDataset(torch.utils.data.Dataset):
    def __init__(self, model_n_feature, tokenizer):
        self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=model_n_feature)))
        self.tokenizer = tokenizer

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()
        feats, feat_lens = self.extractor(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        return {"inputs_padded": feats, "input_lengths": feat_lens, "labels_padded": tokens, "label_lengths": token_lens}
