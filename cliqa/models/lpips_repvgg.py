import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from nunif.models import Model, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from nunif.utils.repvgg import B1_FEATURE_CHANNELS, B1_FEATURE_NODES, create_RepVGG_B1


def spatial_average(x, keepdim=True):
    return x.mean([-2, -1], keepdim=keepdim)


def upsample(x, out_HW=(64, 64)):
    return F.interpolate(x, size=out_HW, mode="bilinear", align_corners=False)


def non_negative_constraint(conv):
    nn.init.constant_(conv.weight, (1.0 / conv.weight.shape[1]))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
    parametrize.register_parametrization(conv, "weight", nn.Softplus())
    return conv


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
        lins = []
        for i in range(self.L):
            non_negative_conv = nn.Conv2d(
                B1_FEATURE_CHANNELS[selected_layers[i]], 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            non_negative_conv = non_negative_constraint(non_negative_conv)
            lins.append(
                nn.Sequential(
                    nn.Dropout(p=0.5),
                    non_negative_conv,
                )
            )

        self.lins = nn.ModuleList(lins)
        self.dist2logits_mlp = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
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
    def forward(self, input, target, return_per_layer=False, spatial=False):
        feats0 = list(self.feature_extractor(self.preprocess(input)).values())
        feats1 = list(self.feature_extractor(self.preprocess(target)).values())

        val = 0
        res = []
        for i in range(self.L):
            f0 = F.normalize(feats0[i].float(), p=2, dim=1, eps=1e-5)
            f1 = F.normalize(feats1[i].float(), p=2, dim=1, eps=1e-5).detach()
            diff = (f0 - f1) ** 2
            dist = self.lins[i](diff)
            if spatial:
                d = upsample(dist)
            else:
                d = spatial_average(dist)

            val = val + d
            if return_per_layer:
                res.append(d)

        if return_per_layer:
            return val, res

        return val


def _test():
    model = LPIPSRepVGG().cuda()
    input = torch.rand((4, 3, 64, 64)).cuda()
    target = torch.rand((4, 3, 64, 64)).cuda()

    z = model(input, target)
    print(z.shape)


if __name__ == "__main__":
    _test()
