import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Mapping, Optional, Tuple


class ConvNormAct(nn.Module):
    """
    Standard Conv-BN-Activation block.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: bool | nn.Module = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """
    Lightweight channel reweighting used in adaptive refinement.
    """

    def __init__(self, channels: int, rd_ratio: float = 0.25):
        super().__init__()
        rd_channels = max(1, int(channels * rd_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class DetailPreservingSpatialRearrangement(nn.Module):
    """
    Detail-Preserving Spatial Rearrangement.

    Corresponds to the upper branch in the CAFM figure:
    Space-to-channel slicing -> Concat -> Dilated 3x3 Conv -> 1x1 Conv.

    It preserves compact fire-detail cues by rearranging interleaved spatial
    samples into channel space. The raw output has half spatial resolution, and
    CAFM upsamples it back before final fusion when needed.
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None, dilation: int = 2):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = in_channels * 4

        self.context_enhancement = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.channel_projection = ConvNormAct(hidden_channels, out_channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  [B, C, H, W]
        # Output: [B, C_out, H/2, W/2]
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            # Keep slicing safe for odd feature sizes.
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")

        x = torch.cat(
            [
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ],
            dim=1,
        )
        x = self.context_enhancement(x)
        return self.channel_projection(x)


class DualBranchContextModeling(nn.Module):
    """
    Dual-Branch Context Modeling.

    Corresponds to the middle branch in the CAFM figure:
    - Global Context Branch for diffuse smoke-context responses.
    - Local Detail Branch for compact small-fire detail responses.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0, "channels must be larger than the grouping factor."

        group_channels = channels // self.groups

        # Global context branch: coordinate-aware context interaction.
        self.global_context_projection = nn.Conv2d(group_channels, group_channels, kernel_size=1, bias=False)
        self.global_context_bn = nn.BatchNorm2d(group_channels)

        # Local detail branch.
        self.local_detail_conv = nn.Conv2d(
            group_channels,
            group_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=group_channels,
            bias=False,
        )
        self.local_detail_bn = nn.BatchNorm2d(group_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        b, c, h, w = x.size()

        # Group processing for efficient context modeling.
        group_x = x.reshape(b * self.groups, -1, h, w)

        # Global Context Branch.
        x_h = group_x.mean(dim=3, keepdim=True)                       # [B*g, Cg, H, 1]
        x_w = group_x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)    # [B*g, Cg, W, 1]
        global_context = torch.cat([x_h, x_w], dim=2)
        global_context = self.global_context_projection(global_context)
        global_context = self.global_context_bn(global_context)

        x_h, x_w = torch.split(global_context, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        global_context_weight = x_h * x_w

        # Local Detail Branch.
        local_detail_weight = self.local_detail_conv(group_x)
        local_detail_weight = self.local_detail_bn(local_detail_weight)

        combined_weight = self.sigmoid(global_context_weight + local_detail_weight)
        out = group_x * combined_weight
        out = out.reshape(b, c, h, w)

        if return_weights:
            global_context_weight = global_context_weight.reshape(b, c, h, w)
            local_detail_weight = local_detail_weight.reshape(b, c, h, w)
            return out, global_context_weight, local_detail_weight
        return out


class OrthogonalSpatialBranch(nn.Module):
    """
    Orthogonal Spatial Branch.

    Corresponds to the right-top branch in the CAFM figure:
    1xK, Kx1, KxK and 1x1 depthwise convolutions capture horizontal, vertical,
    large-area and compact local spatial responses.
    """

    def __init__(self, dim: int, kernel_size: int = 31):
        super().__init__()
        pad = kernel_size // 2

        self.horizontal_context = nn.Conv2d(
            dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), groups=dim, bias=True
        )
        self.vertical_context = nn.Conv2d(
            dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=dim, bias=True
        )
        self.large_area_context = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=True
        )
        self.compact_local_response = nn.Conv2d(
            dim, dim, kernel_size=1, padding=0, groups=dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.horizontal_context(x)
            + self.vertical_context(x)
            + self.large_area_context(x)
            + self.compact_local_response(x)
        )


class ContextGuidedAttention(nn.Module):
    """
    Context-guided spatial/channel attention.

    This module replaces harmonic/spectral attention terminology with a plain
    RGB-context-guided attention mechanism.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.spatial_projection = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weight = self.sigmoid(self.channel_projection(self.channel_pool(x)))
        spatial_weight = self.spatial_projection(x)
        return x * channel_weight * spatial_weight


class ContextGuidedCrossScaleRefinement(nn.Module):
    """
    Context-Guided Cross-Scale Refinement.

    Corresponds to the lower part of Cross-Scale Spatial Modeling in the CAFM
    figure: coarse/medium/fine context aggregation followed by spatial attention.

    In a single-input CAFM block, the three scales are approximated by pooled
    contextual views of the same feature map. If external multi-scale features are
    available, pass them through `multi_scale_features`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale_fusion = ConvNormAct(dim * 3, dim, k=1)
        self.local_refine = ConvNormAct(dim, dim, k=3, g=dim)
        self.context_attention = ContextGuidedAttention(dim)

    def forward(
        self,
        x: torch.Tensor,
        multi_scale_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h, w = x.shape[-2:]

        if multi_scale_features is None:
            fine = x
            medium = F.interpolate(
                F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            coarse = F.interpolate(
                F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=True),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            coarse, medium, fine = multi_scale_features
            coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
            medium = F.interpolate(medium, size=(h, w), mode="bilinear", align_corners=False)
            fine = F.interpolate(fine, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([coarse, medium, fine], dim=1)
        x = self.scale_fusion(x)
        x = self.local_refine(x)
        return self.context_attention(x)


class CrossScaleSpatialModeling(nn.Module):
    """
    Cross-Scale Spatial Modeling.

    Corresponds to the right branch in the CAFM figure:
    Orthogonal Spatial Branch + Context-Guided Cross-Scale Refinement.
    """

    def __init__(self, dim: int, kernel_size: int = 31):
        super().__init__()
        self.input_projection = ConvNormAct(dim, dim, k=1, act=nn.GELU())
        self.orthogonal_spatial_branch = OrthogonalSpatialBranch(dim, kernel_size=kernel_size)
        self.context_guided_refinement = ContextGuidedCrossScaleRefinement(dim)
        self.output_projection = ConvNormAct(dim, dim, k=1)

    def forward(
        self,
        x: torch.Tensor,
        multi_scale_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        base = self.input_projection(x)
        orthogonal_response = self.orthogonal_spatial_branch(base)
        refined_context = self.context_guided_refinement(base, multi_scale_features=multi_scale_features)
        return self.output_projection(base + orthogonal_response + refined_context)


class AdaptiveGlobalLocalRefinement(nn.Module):
    """
    Adaptive Global-Local Refinement.

    Corresponds to the bottom branch in the CAFM figure:
    Global-local context generation -> Local Feature Extraction -> SE Block
    -> 1x1 projection -> residual refinement.
    """

    def __init__(
        self,
        dim: int,
        se_ratio: float = 0.25,
        drop: float = 0.0,
    ):
        super().__init__()
        self.global_local_generator = DualBranchContextModeling(dim)
        self.local_feature_extraction = ConvNormAct(dim, dim, k=3, g=dim)
        self.channel_reweighting = SqueezeExcite(dim, rd_ratio=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.output_projection = ConvNormAct(dim, dim, k=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.global_local_generator(x)
        x = self.local_feature_extraction(x)
        x = self.channel_reweighting(x)
        x = self.proj_drop(x)
        x = self.output_projection(x)

        return shortcut + x


class CAFM(nn.Module):
    """
    CAFM: Context-Aware Feature Fusion Module.

    This version is aligned with the CAFM architecture figure. It contains four
    explicit branches:

    1. Detail-Preserving Spatial Rearrangement
    2. Dual-Branch Context Modeling
    3. Cross-Scale Spatial Modeling
    4. Adaptive Global-Local Refinement

    Their outputs are concatenated and projected by a 1x1 convolution to produce
    the final context-aware fused feature.
    """

    def __init__(
        self,
        dim: int,
        reduction_ratio: float = 1.0,
        context_groups: int = 8,
        spatial_kernel_size: int = 31,
        se_ratio: float = 0.25,
        use_residual: bool = True,
    ):
        super().__init__()
        hidden_dim = max(1, int(dim * reduction_ratio))
        self.use_residual = use_residual and (hidden_dim == dim)

        self.input_projection = ConvNormAct(dim, hidden_dim, k=1)

        self.spatial_rearrangement = DetailPreservingSpatialRearrangement(hidden_dim, hidden_dim)
        self.dual_branch_context = DualBranchContextModeling(hidden_dim, factor=context_groups)
        self.cross_scale_modeling = CrossScaleSpatialModeling(hidden_dim, kernel_size=spatial_kernel_size)
        self.adaptive_refinement = AdaptiveGlobalLocalRefinement(hidden_dim, se_ratio=se_ratio)

        self.branch_fusion = ConvNormAct(hidden_dim * 4, dim, k=1)

    def forward(
        self,
        x: torch.Tensor,
        multi_scale_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        shortcut = x
        h, w = x.shape[-2:]

        x = self.input_projection(x)

        # Branch 1: detail-preserving spatial rearrangement.
        spatial_rearranged = self.spatial_rearrangement(x)
        spatial_rearranged = F.interpolate(
            spatial_rearranged,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Branch 2: dual-branch context modeling.
        context_feature = self.dual_branch_context(x)

        # Branch 3: cross-scale spatial modeling.
        cross_scale_feature = self.cross_scale_modeling(x, multi_scale_features=multi_scale_features)

        # Branch 4: adaptive global-local refinement.
        refined_feature = self.adaptive_refinement(x)

        out = torch.cat(
            [spatial_rearranged, context_feature, cross_scale_feature, refined_feature],
            dim=1,
        )
        out = self.branch_fusion(out)

        if self.use_residual:
            out = out + shortcut
        return out


__all__ = [
    "CAFM",
    "DetailPreservingSpatialRearrangement",
    "DualBranchContextModeling",
    "CrossScaleSpatialModeling",
    "OrthogonalSpatialBranch",
    "ContextGuidedCrossScaleRefinement",
    "ContextGuidedAttention",
    "AdaptiveGlobalLocalRefinement",
    "ConvNormAct",
    "SqueezeExcite",
]


if __name__ == "__main__":
    print("Testing HAG-Net CAFM implementation...")

    input_tensor = torch.randn(2, 64, 64, 64)
    model = CAFM(dim=64, reduction_ratio=1.0, context_groups=8, spatial_kernel_size=31)

    output = model(input_tensor)

    print(f"Input:  {input_tensor.shape}")
    print(f"Output: {output.shape}")

    # Optional multi-scale feature test for Context-Guided Cross-Scale Refinement.
    coarse = torch.randn(2, 64, 16, 16)
    medium = torch.randn(2, 64, 32, 32)
    fine = torch.randn(2, 64, 64, 64)
    output_ms = model(input_tensor, multi_scale_features=(coarse, medium, fine))

    print(f"Output with external multi-scale features: {output_ms.shape}")
    print("HAG-Net CAFM forward test passed.")


