import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones import *
from model.heads import *


class SegmentNet(nn.Module):
    def __init__(self, backbone: str = "MiT-B0", num_classes: int = 19):
        super().__init__()

        self.backbone = MiT(backbone)
        self.params = self.backbone.channels

        self.decoder = SegFormerHead(
            self.backbone.channels,
            256 if "B0" in backbone or "B1" in backbone else 768,
            num_classes,
        )

    def forward(self, x):
        y = self.backbone(x)
        global_mask = self.decoder(y)
        global_mask = F.interpolate(
            global_mask, size=x.size()[2:], mode="bilinear", align_corners=False
        )

        return global_mask


from thop import profile
from thop import clever_format


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("[Statistics Information]\nFLOPs: {}\nParams: {}".format(flops, params))


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    CalParams(model, x)

    outs, out = model(x, False)

    print(out.shape)
