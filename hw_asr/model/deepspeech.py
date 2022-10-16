from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

# Don't pay attention to the name, initially I wanted to implement DeepSpeech...
class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden=512, num_layers=4, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.lstm = nn.LSTM(input_size=n_feats, hidden_size=hidden,bidirectional=True, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden*2),
            nn.Linear(hidden*2, n_class, bias=False)
        )
        
    def forward(self, spectrogram, **batch):
        inp = spectrogram.transpose(1, 2).transpose(0, 1).contiguous()
        out, _ = self.lstm(inp)
        length, batch_size = out.size(0), out.size(1)
        out = out.view(length * batch_size, -1)
        out = self.fc(out)
        out = out.view(length, batch_size, -1)
        return {"logits": out.transpose(0, 1)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here