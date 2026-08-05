import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.models import Model, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import RMSNorm2d
from nunif.utils.repvgg import RepVGG_B1


class Dist2Logit(nn.Module):
    def __init__(self, num_kernels=8):
        super().__init__()
        self.num_kernels = num_kernels
        self.gammas = nn.Parameter(
            torch.cat((torch.linspace(-2.0, 2.0, num_kernels), torch.linspace(-2.0, 2.0, num_kernels)), dim=0).view(
                1, num_kernels * 2, 1, 1
            )
        )
        self.weights = nn.Parameter(torch.zeros(num_kernels * 2).view(1, num_kernels * 2, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def additive_model(x, gammas, weights):
        gammas = F.softplus(gammas)
        weights = F.softplus(weights)
        bases = torch.tanh(x * gammas)
        output = (bases * weights).sum(dim=1, keepdim=True)
        return output

    def forward(self, d0, d1):
        x1 = d0 - d1
        x2 = torch.log(d0 + 1e-5) - torch.log(d1 + 1e-5)
        g1, g2 = self.gammas.chunk(2, dim=1)
        w1, w2 = self.weights.chunk(2, dim=1)
        logits = self.additive_model(x1, g1, w1) + self.additive_model(x2, g2, w2) + self.bias
        return logits


@register_model
class LPIPSRepVGG(Model):
    name = "cliqa.lpips_repvgg"

    def __init__(self):
        super().__init__({})
        selected_layers = ["l4", "l8"]  # "l4", "l8", "l16_h", "l16"
        self.L = len(selected_layers)
        nodes = {}
        for layer_name, name in RepVGG_B1.FEATURE_NODES.items():
            if name in selected_layers:
                nodes[layer_name] = name
        self.feature_extractor = RepVGG_B1.from_pretrained().create_feature_extractor(nodes)
        self.feature_extractor.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.rms_norm = nn.ModuleList(
            [RMSNorm2d(RepVGG_B1.FEATURE_CHANNELS[selected_layers[i]]) for i in range(self.L)]
        )

        # This module is called by the trainer
        self.dist2logits = Dist2Logit()

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.train(False)
        self.feature_extractor.requires_grad_(False)
        return self

    def preprocess(self, x):
        if x.ndim == 4:
            return (x - self.mean) / self.std
        else:
            return (x - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, input, target):
        feats0 = list(self.feature_extractor(self.preprocess(input)).values())
        feats1 = list(self.feature_extractor(self.preprocess(target)).values())

        with torch.autocast(device_type=input.device.type, enabled=False):
            val = 0
            for i in range(self.L):
                f0 = self.rms_norm[i](feats0[i].float())
                f1 = self.rms_norm[i](feats1[i].float()).detach()
                diff = (f0 - f1) ** 2
                diff = F.dropout(diff, p=0.5, training=self.training)
                dist = diff.mean(dim=[1, 2, 3], keepdim=True)
                val = val + dist

            return val


def _test():
    model = LPIPSRepVGG().cuda()
    input = torch.rand((4, 3, 64, 64)).cuda()
    target = torch.rand((4, 3, 64, 64)).cuda()

    z = model(input, target)
    print(z.shape)


if __name__ == "__main__":
    _test()
