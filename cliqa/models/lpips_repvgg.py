import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.models import Model, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from nunif.modules.norm import RMSNorm2d
from nunif.utils.repvgg import B1_FEATURE_CHANNELS, B1_FEATURE_NODES, create_RepVGG_B1


@register_model
class LPIPSRepVGG(Model):
    name = "cliqa.lpips_repvgg"

    def __init__(self):
        super().__init__({})
        selected_layers = ["l4", "l8"]  # "l4", "l8", "l16_h", "l16"
        self.L = len(selected_layers)
        nodes = {}
        for layer_name, name in B1_FEATURE_NODES.items():
            if name in selected_layers:
                nodes[layer_name] = name
        self.feature_extractor = create_RepVGG_B1().create_feature_extractor(nodes)
        self.feature_extractor.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.rms_norm = nn.ModuleList([RMSNorm2d(B1_FEATURE_CHANNELS[selected_layers[i]]) for i in range(self.L)])
        self.dist2logits_mlp = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )
        basic_module_init(self.dist2logits_mlp)

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
    def dist2logits(self, d0, d1, eps=0.1):
        x = torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1)
        return self.dist2logits_mlp(x)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, input, target):
        feats0 = list(self.feature_extractor(self.preprocess(input)).values())
        feats1 = list(self.feature_extractor(self.preprocess(target)).values())

        with torch.autocast(device_type=input.device.type, enabled=False):
            val = 0
            for i in range(self.L):
                f0 = self.rms_norm[i](feats0[i].float())
                f1 = self.rms_norm[i](feats1[i].float())
                diff = (f0 - f1) ** 2
                diff = F.dropout(diff, p=0.5, training=self.training)
                dist = diff.mean(dim=1, keepdim=True)
                spatial_average = dist.mean([-2, -1], keepdim=True)
                val = val + spatial_average

            return val


def _test():
    model = LPIPSRepVGG().cuda()
    input = torch.rand((4, 3, 64, 64)).cuda()
    target = torch.rand((4, 3, 64, 64)).cuda()

    z = model(input, target)
    print(z.shape)


if __name__ == "__main__":
    _test()
