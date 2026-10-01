import torch
import torch.nn as nn
from typing import Dict, Mapping, Optional, Tuple


class DropPath(nn.Module):
    """
    Stochastic depth per sample.

    This fallback keeps the module self-contained. If your project already uses
    DropPath from another package, you can replace this implementation directly.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBNAct(nn.Module):
    """
    Standard Conv-BN-SiLU projection for lightweight channel mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ContextGateCore(nn.Module):
    """
    Context gate core used inside ACG.

    The input channels are split into two complementary branches:
    - local branch: captures compact visual details, local texture and boundary cues;
    - context branch: captures enlarged contextual responses using dilated depthwise convolution.

    This module is designed for RGB UAV feature maps and does not assume thermal,
    radiometric, spectral, or frequency-domain input information.
    """

    def __init__(self, dim: int, dilation: int = 3):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"ContextGateCore expects an even channel number, got dim={dim}.")

        self.dim_half = dim // 2

        self.local_path = nn.Conv2d(
            self.dim_half,
            self.dim_half,
            kernel_size=3,
            padding=1,
            groups=self.dim_half,
            bias=True,
        )

        self.context_path = nn.Conv2d(
            self.dim_half,
            self.dim_half,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=self.dim_half,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat, context_feat = torch.split(x, [self.dim_half, self.dim_half], dim=1)
        local_feat = self.local_path(local_feat)
        context_feat = self.context_path(context_feat)
        return torch.cat([local_feat, context_feat], dim=1)


class ACG(nn.Module):
    """
    ACG: Adaptive Context Gating.

    HAG-Net structure:
    Input -> Linear -> Value/Gate split
                  Gate -> Conv 3x3 + Dilated Conv 3x3 -> Concat -> Sigmoid
    Value * Gate -> Linear -> Output

    ACG adaptively gates the value branch with local-detail and enlarged-context
    responses extracted from RGB UAV feature maps. It is used in DMMM to enhance
    discriminative morphology, texture and boundary cues.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # GLU-style bottleneck. The gate dimension is kept even because
        # ContextGateCore splits channels into two equal branches.
        hidden_features = int(2 * hidden_features / 3)
        if hidden_features % 2 != 0:
            hidden_features += 1

        self.use_residual = use_residual and (in_features == out_features)

        self.input_projection = nn.Linear(in_features, hidden_features * 2)
        self.context_gate = ContextGateCore(hidden_features)

        # Use Sigmoid here to match the ACG figure.
        self.gate_activation = nn.Sigmoid()

        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # NCHW -> NHWC for linear projection.
        x = x.permute(0, 2, 3, 1)

        x_proj = self.input_projection(x)
        x_value, x_gate = x_proj.chunk(2, dim=-1)

        # Gate branch: NHWC -> NCHW for depthwise context modeling.
        x_gate = x_gate.permute(0, 3, 1, 2)
        x_gate = self.context_gate(x_gate)
        x_gate = self.gate_activation(x_gate)
        x_gate = x_gate.permute(0, 2, 3, 1)

        # Adaptive context gating.
        x = x_value * x_gate

        x = self.dropout(x)
        x = self.output_projection(x)
        x = self.dropout(x)

        # NHWC -> NCHW.
        x = x.permute(0, 3, 1, 2)

        if self.use_residual:
            x = x + shortcut
        return x


class DMKDC(nn.Module):
    """
    DMKDC: Directional Multi-Kernel Dynamic Convolution.

    HAG-Net branch design:
    - K x K branch: local square-context response;
    - 1 x K branch: horizontal directional response;
    - K x 1 branch: vertical directional response.

    Direction weights are generated from the input feature through global average
    pooling and a 1x1 convolution for adaptive branch fusion.
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_branches = 3

        # When band_kernel_size is not specified, use the same K as the square
        # branch so the implementation exactly matches the figure: KxK, 1xK, Kx1.
        band_kernel_size = band_kernel_size or square_kernel_size

        self.square_kernel_size = square_kernel_size
        self.band_kernel_size = band_kernel_size

        self.dwconv = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=square_kernel_size,
                padding=square_kernel_size // 2,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
                groups=in_channels,
                bias=True,
            ),
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        self.direction_weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * self.num_branches, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape

        direction_weights = self.direction_weight_generator(x)
        direction_weights = direction_weights.view(batch_size, self.num_branches, channels, 1, 1)
        direction_weights = torch.softmax(direction_weights, dim=1)

        branch_outputs = []
        for i, conv in enumerate(self.dwconv):
            branch_outputs.append(conv(x) * direction_weights[:, i])

        x = torch.stack(branch_outputs, dim=0).sum(dim=0)
        return self.act(self.bn(x))


class DirectionalMorphologyMixer(nn.Module):
    """
    Directional morphology mixer used inside DMMM.

    The input channels are split into two groups. Each group is processed by a
    DMKDC with a different base kernel size:
    - DMKDC(k=3): 3x3, 1x3 and 3x1 depthwise branches;
    - DMKDC(k=5): 5x5, 1x5 and 5x1 depthwise branches.

    The outputs are concatenated and projected to the original channel dimension.
    """

    def __init__(self, channel: int = 256, kernels: Tuple[int, int] = (3, 5)):
        super().__init__()
        if channel % len(kernels) != 0:
            raise ValueError(
                f"DirectionalMorphologyMixer expects channels divisible by the number of kernels: "
                f"channel={channel}, kernels={kernels}."
            )

        self.groups = len(kernels)
        group_channels = channel // self.groups

        self.dmkdc_branches = nn.ModuleList([
            DMKDC(group_channels, square_kernel_size=ks, band_kernel_size=ks)
            for ks in kernels
        ])

        self.projection = ConvBNAct(channel, channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_groups = torch.chunk(x, chunks=self.groups, dim=1)
        x_groups = [self.dmkdc_branches[i](x_groups[i]) for i in range(self.groups)]
        x = torch.cat(x_groups, dim=1)
        return self.projection(x)


class DMMM(nn.Module):
    """
    DMMM: Directional Morphology Modeling Module.

    HAG-Net structure:
    Norm -> Split -> DMKDC(k=3) / DMKDC(k=5) -> Concat -> Conv -> Add
    Norm -> ACG -> Add

    The module is formulated for RGB UAV fire and smoke detection, emphasizing
    visible-image morphology, texture, boundary and contextual cues.
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.morphology_mixer = DirectionalMorphologyMixer(dim)
        self.acg = ACG(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(
            self.layer_scale_1.view(1, -1, 1, 1) * self.morphology_mixer(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.view(1, -1, 1, 1) * self.acg(self.norm2(x))
        )
        return x


__all__ = [
    "DMMM",
    "DMKDC",
    "DirectionalMorphologyMixer",
    "ACG",
    "ContextGateCore",
    "ConvBNAct",
    "DropPath",
]


if __name__ == "__main__":
    input_tensor = torch.randn(2, 128, 64, 64)

    model = DMMM(dim=128)
    output_tensor = model(input_tensor)

    print(f"Input Shape:  {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")

    for idx, branch in enumerate(model.morphology_mixer.dmkdc_branches):
        print(
            f"Branch {idx}: square K={branch.square_kernel_size}, "
            f"directional K={branch.band_kernel_size}"
        )

    print("HAG-Net DMMM forward test passed.")


