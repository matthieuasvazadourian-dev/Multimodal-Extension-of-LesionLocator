"""
Intermediate feature-level fusion modules for PET+CT segmentation.

Three variants:
  - TAMWLiteFusion  (Variant A): channel-gated affine weighting, no decoder dependency
  - MCSAFusion      (Variant B): windowed bidirectional cross-attention (BDSA)
  - CombinedFusion  (Variant C): MCSA -> TAMWLite in sequence

All modules are designed to:
  - Accept (skip_ct, skip_pet) tensors of shape [B, C, D, H, W]
  - Output a fused tensor of the same C channels (decoder-compatible)
  - Init to CT-passthrough so model == CT-only at epoch 0

SimpleConcatFusion is used at shallow encoder levels (k < 3) where MCSA is too expensive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Utility: Simple concat + 1x1 projection (shallow levels / fallback)
# ---------------------------------------------------------------------------

class SimpleConcatFusion(nn.Module):
    """Concat CT and PET skips, project back to C channels. CT-passthrough init."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1, bias=False)
        self._init_ct_passthrough(in_channels)

    def _init_ct_passthrough(self, C: int):
        # Weight shape: [C_out, 2*C_in, 1, 1, 1]
        # First C_in input channels = CT, last C_in = PET
        # Init: identity on CT channels, zero on PET channels
        with torch.no_grad():
            self.proj.weight.zero_()
            for i in range(C):
                self.proj.weight[i, i] = 1.0

    def forward(self, skip_ct: torch.Tensor, skip_pet: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([skip_ct, skip_pet], dim=1))


# ---------------------------------------------------------------------------
# Variant A: TAMW-lite (channel-gating without prediction maps)
# ---------------------------------------------------------------------------

class TAMWLiteFusion(nn.Module):
    """
    Channel-wise affine gating of concatenated [CT, PET] skip features.

    Procedure (per H2ASeg TAMW spirit, no decoder prediction maps):
      1. concat [s_ct, s_pet] -> F in R^{B x 2C x D x H x W}
      2. global average pool -> R^{B x 2C}
      3. MLP + tanh -> gates W in R^{B x 2C}  (range [-1, 1])
      4. F_enh = W * F  (per-channel rescaling, broadcast)
      5. 1x1x1 conv: 2C -> C  (CT-passthrough init)

    CT-passthrough init: MLP bias = 0 -> tanh(0) = 0 -> gates = 0
    -> F_enh = 0 * F = 0, then proj reproduces CT channels from identity.
    To achieve epoch-0 CT-only behaviour:
      - gates initialized to [1 ... 1, 0 ... 0] (CT channels=1, PET channels=0)
      - proj initialized to identity on concatenated input (only CT half matters)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels
        self.gap = nn.AdaptiveAvgPool3d(1)
        # MLP: 2C -> 4C -> 2C, tanh activation
        self.mlp = nn.Sequential(
            nn.Linear(2 * C, 4 * C, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4 * C, 2 * C, bias=True),
            nn.Tanh(),
        )
        self.proj = nn.Conv3d(2 * C, C, kernel_size=1, bias=False)
        self._init_ct_passthrough(C)

    def _init_ct_passthrough(self, C: int):
        with torch.no_grad():
            # proj: identity on CT channels (first C of 2C input), zero on PET channels.
            # This gives near-CT-passthrough at epoch 0:
            #   output = skip_ct + proj(gates * [skip_ct, skip_pet])
            #          ≈ skip_ct + proj(tiny * [skip_ct, skip_pet])   (since gates ≈ 0)
            #          ≈ skip_ct + tiny * skip_ct  ≈  skip_ct
            # Crucially, proj.weight != 0 so gradients flow from the start.
            self.proj.weight.zero_()
            for i in range(C):
                self.proj.weight[i, i] = 1.0   # identity on CT half only
            # MLP: small-scale random init (NOT zero) to avoid zero-gradient dead-start.
            # With zero MLP weights, gates=0 -> F_enh=0 -> proj sees zero input
            # -> dL/d(proj) = 0 -> dL/d(gates) = 0 -> dL/d(MLP) = 0. Module stuck.
            # Small std keeps initial output ≈ skip_ct while allowing gradient flow.
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, skip_ct: torch.Tensor, skip_pet: torch.Tensor) -> torch.Tensor:
        F_cat = torch.cat([skip_ct, skip_pet], dim=1)           # [B, 2C, D, H, W]
        gap = self.gap(F_cat).flatten(1)                         # [B, 2C]
        gates = self.mlp(gap)                                    # [B, 2C]  in [-1,1]
        gates = gates.view(*gates.shape, 1, 1, 1)               # [B, 2C, 1, 1, 1]
        F_enh = gates * F_cat                                    # [B, 2C, D, H, W]
        # Residual: skip_ct + learned modulation
        return skip_ct + self.proj(F_enh)                        # [B, C, D, H, W]


# ---------------------------------------------------------------------------
# Variant B: MCSA — Windowed Bidirectional Spatial Attention
# ---------------------------------------------------------------------------

def _pad_to_divisible_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Pad spatial dims to next multiple of window_size. Returns padded tensor and pad amounts."""
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    wd, wh, ww = window_size
    pad_d = (wd - D % wd) % wd
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    return x, (pad_d, pad_h, pad_w)


def _unpad_3d(x: torch.Tensor, pads: Tuple[int, int, int]) -> torch.Tensor:
    pad_d, pad_h, pad_w = pads
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    return x[:, :,
             :D - pad_d if pad_d > 0 else D,
             :H - pad_h if pad_h > 0 else H,
             :W - pad_w if pad_w > 0 else W]


def _window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition into non-overlapping windows.
    x: [B, C, D, H, W]
    Returns: [num_windows * B, C, wd, wh, ww]
    """
    B, C, D, H, W = x.shape
    wd, wh, ww = window_size
    x = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()   # [B, nD, nH, nW, C, wd, wh, ww]
    x = x.view(-1, C, wd, wh, ww)
    return x


def _window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int],
                        D: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    windows: [num_windows * B, C, wd, wh, ww]
    Returns: [B, C, D, H, W]
    """
    wd, wh, ww = window_size
    nD, nH, nW = D // wd, H // wh, W // ww
    B = windows.shape[0] // (nD * nH * nW)
    C = windows.shape[1]
    x = windows.view(B, nD, nH, nW, C, wd, wh, ww)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()   # [B, C, nD, wd, nH, wh, nW, ww]
    x = x.view(B, C, D, H, W)
    return x


class BDSABlock(nn.Module):
    """
    Bidirectional Spatial Attention (BDSA) from H2ASeg (eq. 1).

    Given source s and target t (both shape [N, C, ...] where N = num_windows*B):
      A_self  = V_s  * softmax((Q_t^T K_s) / sqrt(C))^T   <- cross: t queries s
      A_cross = V_t  * softmax((Q_t^T K_t) / sqrt(C))^T   <- self:  t queries t

    Enhanced target = 3x3x3_conv(concat([A_self, A_cross]))

    Run twice (PET->CT and CT->PET) with shared projections.

    Projection is done via 1x1x1 convs on spatial tokens (flattened over spatial dims).
    """

    def __init__(self, channels: int):
        super().__init__()
        C = channels
        # Shared projections — used for both source and target roles
        self.proj_q = nn.Conv3d(C, C, kernel_size=1, bias=False)
        self.proj_k = nn.Conv3d(C, C, kernel_size=1, bias=False)
        self.proj_v = nn.Conv3d(C, C, kernel_size=1, bias=False)
        self.mixer  = nn.Conv3d(2 * C, C, kernel_size=3, padding=1, bias=False)
        self.scale  = C ** -0.5
        # mixer near-zero init -> enhanced feature ≈ 0, residual carries signal
        nn.init.zeros_(self.mixer.weight)

    def _attend(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        source, target: [N, C, wd, wh, ww]
        Returns (A_self, A_cross) each of shape [N, C, wd, wh, ww]
          A_self  = cross-attention: target queries source (target gets source context)
          A_cross = self-attention: target queries target
        """
        N, C, wd, wh, ww = source.shape
        L = wd * wh * ww

        Q_t = self.proj_q(target).view(N, C, L)   # [N, C, L]
        K_s = self.proj_k(source).view(N, C, L)   # [N, C, L]
        V_s = self.proj_v(source).view(N, C, L)

        K_t = self.proj_k(target).view(N, C, L)
        V_t = self.proj_v(target).view(N, C, L)

        # Attention weights [N, L, L]
        attn_cross = torch.bmm(Q_t.transpose(1, 2), K_s) * self.scale   # [N, L, L]
        attn_cross = attn_cross.softmax(dim=-1)
        attn_self  = torch.bmm(Q_t.transpose(1, 2), K_t) * self.scale
        attn_self  = attn_self.softmax(dim=-1)

        # A_self: target attends to source values
        A_self  = torch.bmm(V_s, attn_cross.transpose(1, 2)).view(N, C, wd, wh, ww)
        # A_cross: target attends to its own values
        A_cross = torch.bmm(V_t, attn_self.transpose(1, 2)).view(N, C, wd, wh, ww)
        return A_self, A_cross

    def forward(self, x_ct: torch.Tensor, x_pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_ct, x_pet: [N, C, wd, wh, ww]  (window tokens)
        Returns DELTA tensors (attention output only, no residual).
        MCSAFusion.forward adds the single outer residual.
        """
        # PET -> CT: CT is target, PET is source
        A_self_ct,  A_cross_ct  = self._attend(source=x_pet, target=x_ct)
        e_ct  = self.mixer(torch.cat([A_self_ct,  A_cross_ct],  dim=1))

        # CT -> PET: PET is target, CT is source
        A_self_pet, A_cross_pet = self._attend(source=x_ct,  target=x_pet)
        e_pet = self.mixer(torch.cat([A_self_pet, A_cross_pet], dim=1))

        return e_ct, e_pet


class MCSAFusion(nn.Module):
    """
    Multi-scale Cross-modal Spatial Attention (MCSA) from H2ASeg.

    Inter-window attention: pool each window to a token, run BDSA across tokens,
      upsample back to original spatial dims.
    Intra-window attention: partition into windows, run BDSA inside each window,
      merge windows back.

    Outputs (e_ct, e_pet) — both enhanced, C channels each.
    Used inside CombinedFusion or directly as a fusion module (returns avg of pair).
    """

    def __init__(self, in_channels: int, window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.window_size = window_size
        # Inter-window BDSA: operates on pooled tokens (spatial = num_windows, C channels)
        self.inter_bdsa = BDSABlock(in_channels)
        # Intra-window BDSA: operates inside each window
        self.intra_bdsa = BDSABlock(in_channels)

    def _inter_attention(self, x_ct: torch.Tensor, x_pet: torch.Tensor,
                          ws: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool to window tokens, attend, upsample back."""
        D, H, W = x_ct.shape[2], x_ct.shape[3], x_ct.shape[4]
        pool_size = (D // ws[0], H // ws[1], W // ws[2])
        t_ct  = F.adaptive_avg_pool3d(x_ct, pool_size)    # [B, C, nD, nH, nW]
        t_pet = F.adaptive_avg_pool3d(x_pet, pool_size)

        e_t_ct, e_t_pet = self.inter_bdsa(t_ct, t_pet)

        # Upsample back to original spatial dims
        e_t_ct  = F.interpolate(e_t_ct,  size=(D, H, W), mode='trilinear', align_corners=False)
        e_t_pet = F.interpolate(e_t_pet, size=(D, H, W), mode='trilinear', align_corners=False)
        return e_t_ct, e_t_pet

    def _intra_attention(self, x_ct: torch.Tensor, x_pet: torch.Tensor,
                          ws: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition into windows, attend inside each window, merge back."""
        x_ct,  pads = _pad_to_divisible_3d(x_ct,  ws)
        x_pet, _    = _pad_to_divisible_3d(x_pet, ws)
        D, H, W = x_ct.shape[2], x_ct.shape[3], x_ct.shape[4]

        w_ct  = _window_partition_3d(x_ct,  ws)   # [nW*B, C, wd, wh, ww]
        w_pet = _window_partition_3d(x_pet, ws)

        e_ct_w, e_pet_w = self.intra_bdsa(w_ct, w_pet)

        e_ct  = _window_reverse_3d(e_ct_w,  ws, D, H, W)
        e_pet = _window_reverse_3d(e_pet_w, ws, D, H, W)

        e_ct  = _unpad_3d(e_ct,  pads)
        e_pet = _unpad_3d(e_pet, pads)
        return e_ct, e_pet

    def forward(self, skip_ct: torch.Tensor, skip_pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ws = self.window_size
        # Clamp window size to actual spatial dims (very shallow levels may be tiny)
        D, H, W = skip_ct.shape[2], skip_ct.shape[3], skip_ct.shape[4]
        ws = (min(ws[0], D), min(ws[1], H), min(ws[2], W))

        e_ct_inter,  e_pet_inter  = self._inter_attention(skip_ct, skip_pet, ws)
        e_ct_intra,  e_pet_intra  = self._intra_attention(skip_ct, skip_pet, ws)

        e_ct  = skip_ct  + e_ct_inter  + e_ct_intra
        e_pet = skip_pet + e_pet_inter + e_pet_intra
        return e_ct, e_pet


# ---------------------------------------------------------------------------
# Variant C: Combined — MCSA -> TAMWLite
# ---------------------------------------------------------------------------

class CombinedFusion(nn.Module):
    """MCSA bidirectional attention followed by TAMWLite channel gating."""

    def __init__(self, in_channels: int, window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.mcsa = MCSAFusion(in_channels, window_size)
        self.tamw = TAMWLiteFusion(in_channels)

    def forward(self, skip_ct: torch.Tensor, skip_pet: torch.Tensor) -> torch.Tensor:
        e_ct, e_pet = self.mcsa(skip_ct, skip_pet)
        return self.tamw(e_ct, e_pet)


# ---------------------------------------------------------------------------
# Wrapper: makes MCSAFusion return a single tensor (averaged pair)
# Used when MCSAFusion is the terminal fusion module (Variant B standalone).
# ---------------------------------------------------------------------------

class MCSAFusionWrapper(nn.Module):
    """Wraps MCSAFusion to return a single fused tensor (mean of enhanced pair)."""

    def __init__(self, in_channels: int, window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.mcsa = MCSAFusion(in_channels, window_size)
        self.proj = nn.Conv3d(2 * in_channels, in_channels, kernel_size=1, bias=False)
        # CT-passthrough init on proj
        with torch.no_grad():
            self.proj.weight.zero_()
            C = in_channels
            for i in range(C):
                self.proj.weight[i, i] = 1.0

    def forward(self, skip_ct: torch.Tensor, skip_pet: torch.Tensor) -> torch.Tensor:
        e_ct, e_pet = self.mcsa(skip_ct, skip_pet)
        return self.proj(torch.cat([e_ct, e_pet], dim=1))
