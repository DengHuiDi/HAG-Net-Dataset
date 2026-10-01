import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p: Optional[int] = None, d: int = 1):
    """
    Pad to keep spatial shape unchanged for stride=1.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_group_norm(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    """
    Create a valid GroupNorm layer even when the channel number is not divisible by 16.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """
    Transform distance representation (left, top, right, bottom) to xywh or xyxy boxes.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)

    return torch.cat((x1y1, x2y2), dim)


def make_anchors(
    feats: Sequence[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points and stride tensors for YOLO-style dense prediction.
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")

        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))

    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)


class ConvGN(nn.Module):
    """
    Conv-GN-SiLU stem used before the gradient-aware localization block.
    """

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None,
                 g: int = 1, d: int = 1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = make_group_norm(c2, max_groups=16)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class IsotropicGradientConv(nn.Module):
    """
    IsoGrad-Conv.

    Central-difference style convolution used to enhance isotropic local gradient
    responses. This is an internal operator of GradientAwareConv.
    """

    def __init__(self, channels: int, theta: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.theta = theta

    def get_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        out_channels, in_channels, k1, k2 = weight.shape
        flat_weight = weight.reshape(out_channels, in_channels, k1 * k2)

        gradient_weight = flat_weight.clone()
        gradient_weight[:, :, 4] = flat_weight[:, :, 4] - self.theta * flat_weight.sum(dim=2)

        gradient_weight = gradient_weight.reshape(out_channels, in_channels, k1, k2)
        return gradient_weight, self.conv.bias


class DiagonalGradientConv(nn.Module):
    """
    DiagGrad-Conv.

    Angular/diagonal difference convolution used to enhance diagonal boundary cues.
    """

    def __init__(self, channels: int, theta: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.theta = theta

    def get_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        out_channels, in_channels, k1, k2 = weight.shape
        flat_weight = weight.reshape(out_channels, in_channels, k1 * k2)

        # Angular difference ordering inherited from the original implementation.
        reordered = flat_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        gradient_weight = flat_weight - self.theta * reordered

        gradient_weight = gradient_weight.reshape(out_channels, in_channels, k1, k2)
        return gradient_weight, self.conv.bias


class HorizontalGradientConv(nn.Module):
    """
    HorizGrad-Conv.

    Builds a 3x3 horizontal-difference kernel from a learnable 1D convolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def get_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        out_channels, in_channels, k = weight.shape

        gradient_weight = torch.zeros(
            out_channels,
            in_channels,
            9,
            dtype=weight.dtype,
            device=weight.device,
        )
        # Left column positive, right column negative.
        gradient_weight[:, :, [0, 3, 6]] = weight
        gradient_weight[:, :, [2, 5, 8]] = -weight

        gradient_weight = gradient_weight.reshape(out_channels, in_channels, k, k)
        return gradient_weight, self.conv.bias


class VerticalGradientConv(nn.Module):
    """
    VertGrad-Conv.

    Builds a 3x3 vertical-difference kernel from a learnable 1D convolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

    def get_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        out_channels, in_channels, k = weight.shape

        gradient_weight = torch.zeros(
            out_channels,
            in_channels,
            9,
            dtype=weight.dtype,
            device=weight.device,
        )
        # Top row positive, bottom row negative.
        gradient_weight[:, :, [0, 1, 2]] = weight
        gradient_weight[:, :, [6, 7, 8]] = -weight

        gradient_weight = gradient_weight.reshape(out_channels, in_channels, k, k)
        return gradient_weight, self.conv.bias


class GradientAwareConv(nn.Module):
    """
    Gradient-aware convolution used by GALB.

    It fuses five operators:
    - IsoGrad-Conv
    - VertGrad-Conv
    - HorizGrad-Conv
    - DiagGrad-Conv
    - standard 3x3 Conv

    The fused response is normalized and activated to strengthen boundary-sensitive
    localization cues for small fire and smoke regions.
    """

    def __init__(self, dim: int, norm: str = "gn", act: Optional[nn.Module] = None):
        super().__init__()

        self.isotropic_gradient = IsotropicGradientConv(dim)
        self.vertical_gradient = VerticalGradientConv(dim)
        self.horizontal_gradient = HorizontalGradientConv(dim)
        self.diagonal_gradient = DiagonalGradientConv(dim)
        self.context_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)

        if norm == "gn":
            self.norm = make_group_norm(dim, max_groups=16)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(dim)
        elif norm in ("none", None):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm type: {norm}")

        self.act = act if act is not None else nn.SiLU()

    def _fused_weight_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w1, b1 = self.isotropic_gradient.get_weight()
        w2, b2 = self.vertical_gradient.get_weight()
        w3, b3 = self.horizontal_gradient.get_weight()
        w4, b4 = self.diagonal_gradient.get_weight()
        w5, b5 = self.context_conv.weight, self.context_conv.bias

        fused_weight = w1 + w2 + w3 + w4 + w5
        fused_bias = b1 + b2 + b3 + b4 + b5

        return fused_weight, fused_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if all(
            hasattr(self, name)
            for name in ["isotropic_gradient", "vertical_gradient", "horizontal_gradient", "diagonal_gradient"]
        ):
            weight, bias = self._fused_weight_bias()
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=1, groups=1)
        else:
            x = self.context_conv(x)

        x = self.norm(x)
        return self.act(x)

    def switch_to_deploy(self):
        """
        Fuse gradient branches into the standard 3x3 convolution for deployment.

        Normalization is intentionally kept outside the fused convolution, matching
        the original behavior before and after branch fusion.
        """
        weight, bias = self._fused_weight_bias()
        self.context_conv.weight = nn.Parameter(weight)
        self.context_conv.bias = nn.Parameter(bias)

        del self.isotropic_gradient
        del self.vertical_gradient
        del self.horizontal_gradient
        del self.diagonal_gradient




class GALB(nn.Module):
    """
    GALB: Gradient-aware Localization Block.

    It enhances boundary-sensitive localization features before prediction.
    """

    def __init__(self, dim: int, num_blocks: int = 2, norm: str = "gn"):
        super().__init__()
        self.blocks = nn.Sequential(
            *[GradientAwareConv(dim, norm=norm) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def switch_to_deploy(self):
        for block in self.blocks:
            if hasattr(block, "switch_to_deploy"):
                block.switch_to_deploy()


GradientAwareLocalizationBlock = GALB


class LearnableScaleLayer(nn.Module):
    """
    Learnable scale layer for bbox regression logits.
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


class GradientAwareDecoupledDetectionHead(nn.Module):
    """
    Gradient-aware Decoupled Detection Head.

    HAG-Net structure:
    P3/P4/P5 -> Stem(1x1 Conv + GN) -> GALB -> decoupled box/class prediction.

    The head enhances boundary-sensitive localization cues through GALB while
    keeping the final prediction branches decoupled for classification and
    bounding-box regression.
    """

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, hidc: int = 256, ch: Sequence[int] = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.stem = nn.ModuleList(nn.Sequential(ConvGN(x, hidc, k=1)) for x in ch)
        self.galb = GALB(hidc, num_blocks=2, norm="gn")

        self.box_pred = nn.Conv2d(hidc, 4 * self.reg_max, kernel_size=1)
        self.cls_pred = nn.Conv2d(hidc, self.nc, kernel_size=1)
        self.scale = nn.ModuleList(LearnableScaleLayer(1.0) for _ in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]):
        """
        Return per-level raw outputs during training, and decoded predictions
        during inference.
        """
        outputs = []

        for i in range(self.nl):
            feat = self.stem[i](x[i])
            feat = self.galb(feat)

            box = self.scale[i](self.box_pred(feat))
            cls = self.cls_pred(feat)

            outputs.append(torch.cat((box, cls), dim=1))

        if self.training:
            return outputs

        shape = outputs[0].shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], dim=2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                tensor.transpose(0, 1) for tensor in make_anchors(outputs, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0) * norm[:, :2],
                xywh=True,
                dim=1,
            )

        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        return y if self.export else (y, outputs)

    def bias_init(self):
        """
        Initialize detection head biases.
        """
        self.box_pred.bias.data[:] = 1.0
        self.cls_pred.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted distance distributions into bounding boxes.
        """
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

    def switch_to_deploy(self):
        """
        Fuse the internal gradient branches in GALB for deployment.
        """
        self.galb.switch_to_deploy()


__all__ = [
    "GradientAwareDecoupledDetectionHead",
    "GALB",
    "GradientAwareLocalizationBlock",
    "GradientAwareConv",
    "IsotropicGradientConv",
    "VerticalGradientConv",
    "HorizontalGradientConv",
    "DiagonalGradientConv",
    "LearnableScaleLayer",
    "ConvGN",
    "DFL",
    "dist2bbox",
    "make_anchors",
]


if __name__ == "__main__":
    print("Testing Gradient-aware Decoupled Detection Head...")

    model = GradientAwareDecoupledDetectionHead(nc=2, hidc=64, ch=(64, 128, 256))
    model.train()

    features = [
        torch.randn(1, 64, 20, 20),
        torch.randn(1, 128, 10, 10),
        torch.randn(1, 256, 5, 5),
    ]

    raw_outputs = model(features)
    print("Training outputs:", [tuple(o.shape) for o in raw_outputs])

    model.eval()
    model.stride = torch.tensor([8.0, 16.0, 32.0])

    with torch.no_grad():
        decoded, raw = model(features)

    print("Inference decoded output:", tuple(decoded.shape))
    print("Raw outputs:", [tuple(o.shape) for o in raw])
    print("Gradient-aware head forward test passed.")


