import torch

import numpy as np

from lhotse import CutSet
from lhotse.dataset.collation import TokenCollater

class LnLTokenCollater(TokenCollater):
    """Collate list of tokens
    Modified version of Lhotse's TokenCollator that makes unk_symbol optional
    See original 
    """

    def __init__(
        self,
        cuts: CutSet,
        # Set all to False by default
        add_eos: bool = False,
        add_bos: bool = False,
        add_unk: bool = False,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
        unk_symbol: str = "<unk>",
    ):
        self.pad_symbol = pad_symbol
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol
        self.unk_symbol = unk_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        tokens = {char for cut in cuts for char in cut.supervisions[0].text}
        tokens_unique = (
            [pad_symbol]
            + ([unk_symbol] if add_unk else [])
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(tokens_unique)}
        self.idx2token = [token for token in tokens_unique]
