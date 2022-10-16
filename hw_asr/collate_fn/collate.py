import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> dict:
    """
    Collate and pad fields in dataset items
    """

    result_batch = {k: [dic[k] for dic in dataset_items] for k in dataset_items[0]}
    
    lengths = []
    for wav in result_batch["audio"]:
        lengths.append(wav.size(-1))

    lengths_spec = []
    for spec in result_batch["spectrogram"]:
        lengths_spec.append(spec.size(-1))
    result_batch["spectrogram_length"] = torch.tensor(lengths_spec)

    text_encoded_length = []
    for text in result_batch["text_encoded"]:
        text_encoded_length.append(text.size(-1))
    result_batch["text_encoded_length"] = torch.tensor(text_encoded_length)

    batch_wavs = torch.zeros(len(dataset_items), max(lengths))
    batch_spec = torch.zeros(len(dataset_items), result_batch["spectrogram"][0].shape[1], max(lengths_spec))
    batch_texts = torch.zeros(len(dataset_items), max(text_encoded_length))
    for i, item in enumerate(dataset_items):
        batch_wavs[i,:lengths[i]] = item["audio"]
        batch_spec[i, :, :lengths_spec[i]] = item["spectrogram"]
        batch_texts[i, :text_encoded_length[i]] = item["text_encoded"]
    
    result_batch["audio"] = batch_wavs
    result_batch["spectrogram"] = batch_spec
    result_batch["text_encoded"] = batch_texts
    
    return result_batch