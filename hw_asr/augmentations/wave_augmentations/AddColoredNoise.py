import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class AddColoredNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)