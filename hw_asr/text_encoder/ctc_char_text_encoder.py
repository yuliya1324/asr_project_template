from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        dp = {
            "": 1.0
        }
        for prob in probs[:probs_length]:
            new_dp = defaultdict(float)
            for string, v in dp.items():
                for i in self.ind2char:
                    ch = self.ind2char[i]
                    if string and ch == string[-1] or ch == self.EMPTY_TOK:
                        new_dp[string] += v * prob[i]
                    else:
                        new_dp[string+self.ind2char[i]] += v * prob[i]
            dp = dict(list(sorted(new_dp.items(), key=lambda x: x[1]))[-beam_size:])
        hypos = [Hypothesis(string, v) for string, v in dp.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]
