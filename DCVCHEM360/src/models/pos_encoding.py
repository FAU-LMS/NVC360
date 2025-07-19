import torch
from torch import nn
import torch.nn.functional as F


class PosEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.channels = 1

    def forward(self, pos):
        weights = torch.cos(pos[:, 1] * 0.5 * torch.pi).clamp(0, 1)
        return weights.unsqueeze(1)


class PosDownsampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pos_enc):
        pos_enc_ds2 = F.interpolate(pos_enc, scale_factor=1 / 2, mode='nearest-exact')
        pos_enc_ds4 = F.interpolate(pos_enc_ds2, scale_factor=1 / 2, mode='nearest-exact')
        pos_enc_ds8 = F.interpolate(pos_enc_ds4, scale_factor=1 / 2, mode='nearest-exact')
        pos_enc_ds16 = F.interpolate(pos_enc_ds8, scale_factor=1 / 2, mode='nearest-exact')
        pos_enc_ds32 = F.interpolate(pos_enc_ds16, scale_factor=1 / 2, mode='nearest-exact')
        pos_enc_ds64 = F.interpolate(pos_enc_ds32, scale_factor=1 / 2, mode='nearest-exact')
        return {
            'ds2': pos_enc_ds2,
            'ds4': pos_enc_ds4,
            'ds8': pos_enc_ds8,
            'ds16': pos_enc_ds16,
            'ds32': pos_enc_ds32,
            'ds64': pos_enc_ds64
        }
