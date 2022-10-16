from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        if "target_text" not in kwargs or "pred_text" not in kwargs:
            raise RuntimeError(f"Beam encoding is not supported by {self.text_encoder}")
        else:
            target_text = kwargs["target_text"]
            pred_text = kwargs["pred_text"]
            wers = []
            for target, pred in zip(target_text, pred_text):
                wers.append(calc_wer(target, pred))
            return sum(wers) / len(wers)