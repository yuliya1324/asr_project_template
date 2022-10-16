from typing import List, NamedTuple
from collections import defaultdict

from pyctcdecode import build_ctcdecoder
import kenlm
import multiprocessing

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCBeamCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, vocabulary, lm, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        with open(vocabulary) as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]
        labels = list(self.ind2char.values())
        labels[0] = ""
        self.decoder = build_ctcdecoder(
            labels,
            lm,
            unigram_list,
        )

    def ctc_decode(self, inds: List[int]) -> str:
        last = self.ind2char[0]
        seq = []
        for i in inds:
            ch = self.ind2char[i]
            if ch != last and i != 0:
                seq.append(ch)
                last = ch
        return "".join(seq)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        beams = self.decoder.decode_beams(probs.cpu().detach().exp().numpy(), beam_width=beam_size)
        return [(beam[0], beam[-2]) for beam in beams]
