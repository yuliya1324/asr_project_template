import random
from copy import copy
import torch
from hw_asr.augmentations.base import AugmentationBase


class SpectAugment(AugmentationBase):
    def __init__(
        self,
        filling_value = "mean",
        n_freq_masks = 2,
        n_time_masks = 2,
        max_freq = 10,
        max_time = 50,
    ):
        self.filling_value = filling_value
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_freq = max_freq
        self.max_time = max_time

    def __call__(self, data):
        sample = copy(data)
        if self.filling_value == "mean":
            v = sample.mean()
        elif self.filling_value == "min":
            v = sample.min()
        elif self.filling_value == "max":
            v = sample.max()
        else:
            v = random.randrange(0, min(self.max_freq, self.max_time))
        mask = torch.full(sample.shape, v)
        for i in range(self.n_time_masks):
            start = random.randrange(0, self.max_time-1)
            end = random.randrange(start+1, self.max_time)
            sample[start:end,:] = mask[start:end,:]
        for i in range(self.n_freq_masks):
            start = random.randrange(0, self.max_freq-1)
            end = random.randrange(start+1, self.max_freq)
            sample[:,start:end] = mask[:,start:end]
        return sample