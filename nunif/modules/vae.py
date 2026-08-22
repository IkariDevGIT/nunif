import torch
import torch.nn as nn


def gaussian_noise(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # NOTE:
    # math.exp(log_var * 0.5) == math.exp(log_var/math.log(math.e**2)) == math.sqrt(math.exp(log_var))
    dtype = mean.dtype
    mean = mean.float()
    log_var = log_var.float()
    standard_deviation = torch.exp(log_var * 0.5)
    noise = torch.randn_like(mean)
    return (mean + (noise * standard_deviation)).to(dtype)


def gaussian_kl_divergence_loss(mean: torch.Tensor, log_var: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    mean = mean.float()
    log_var = log_var.float()
    var = torch.exp(log_var)
    mean2 = mean**2
    kl = -0.5 * (1 + log_var - mean2 - var)
    if reduction == "mean":
        kl = kl.mean()
    elif reduction == "sum":
        # assuming shape[0] is batch dim
        kl = kl.sum()
    return kl


class VAELoss(nn.Module):
    def __init__(self, recon_loss: nn.Module, kld_weight: float = 0.001, return_dict: bool = False):
        super().__init__()
        self.recon_loss = recon_loss
        self.kld_weight = kld_weight
        self.return_dict = return_dict

    def forward(self, input, target) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        input: list[torch.Tensor, torch.Tensor, torch.Tensor] = (Input RGB Image, mean, log_var)
        target: Target RGB Image
        """
        recon, mean, log_var = input
        recon_loss = self.recon_loss(recon, target)
        kld_loss = gaussian_kl_divergence_loss(mean, log_var)
        loss = recon_loss + kld_loss * self.kld_weight

        if self.return_dict:
            return dict(loss=loss, recon_loss=recon_loss, kld_loss=kld_loss)

        return loss
