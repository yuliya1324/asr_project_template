from typing import List, NamedTuple
from fast_ctc_decode import beam_search

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class FastBeamCTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.alphabet = "".join(self.ind2char.values())

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
        seq, path = beam_search(probs.cpu().detach().exp().numpy(), self.alphabet, beam_size=beam_size)
        return [seq, 1]
