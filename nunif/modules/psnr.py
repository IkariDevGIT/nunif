import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PSNR module designed for evaluation.
Note that this is not a loss function for training.
"""


def psnr(input, target):
    mse = F.mse_loss(torch.clamp(input, 0, 1), torch.clamp(target, 0, 1))
    return -10 * torch.log10(1.0 / (mse + 1.0e-6))


def to_luminance(rgb):
    if rgb.shape[1] == 3:
        w = [0.29891, 0.58661, 0.11448]
        return rgb[:, 0:1, :, :] * w[0] + rgb[:, 1:2, :, :] * w[1] + rgb[:, 2:3, :, :] * w[2]
    else:
        assert rgb.shape[1] == 1  # y
        return rgb


class PSNR(nn.Module):
    def forward(self, input, target):
        return psnr(input, target)


class PSNRPerImage(nn.Module):
    def forward(self, input, target):
        if input.ndim == 4:
            psnr_sum = 0
            for x, y in zip(input, target):
                psnr_sum = psnr_sum + psnr(x, y)
            return psnr_sum / input.shape[0]
        else:
            return psnr(input, target)


class LuminancePSNR(nn.Module):
    def forward(self, input, target):
        input = to_luminance(input)
        target = to_luminance(target)
        return psnr(input, target)


class LuminancePSNRPerImage(nn.Module):
    def forward(self, input, target):
        input = to_luminance(input)
        target = to_luminance(target)

        if input.ndim == 4:
            psnr_sum = 0
            for x, y in zip(input, target):
                psnr_sum = psnr_sum + psnr(x, y)
            return psnr_sum / input.shape[0]
        else:
            return psnr(input, target)
