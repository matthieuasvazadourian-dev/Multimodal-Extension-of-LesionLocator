"""
Display learned per-level modality weights of a WeightedSkipFusion checkpoint.

The training-time WeightedSkipFusion applies raw, unconstrained affine weights
(output = alpha_ct * skip_ct + alpha_pet * skip_pet); they are NOT normalized
during the forward pass, so their raw magnitudes are not directly interpretable
as "how much of each modality was used". This tool normalizes them AFTER the
fact, per channel, purely for interpretability:

    w_ct  = |alpha_ct|  / (|alpha_ct| + |alpha_pet|)
    w_pet = |alpha_pet| / (|alpha_ct| + |alpha_pet|)   = 1 - w_ct

and reports, per encoder level, the channel-mean modality split.

Caveat: this is an APPROXIMATE importance measure. alpha also rescales feature
magnitude, and the true per-level contribution depends on the feature norms
||skip_ct||, ||skip_pet|| at inference time — not captured by the weights alone.
It answers "how much did each modality's features get amplified", which is the
interpretability signal requested, not an exact contribution decomposition.

For 'mcsa' checkpoints, only encoder levels below _MCSA_MIN_LEVEL carry scalar
WeightedSkipFusion weights; deeper levels fuse via cross-attention and have no
single scalar weight, so they are reported as "mcsa (attention)".

Usage:
    python -m lesionlocator.utilities.display_fusion_weights --ckpt /path/to/checkpoint_final.pth
"""

import argparse
import re
from collections import defaultdict

import torch


_ALPHA_RE = re.compile(r"(?:^|\.)fusion_modules\.(\d+)\.(alpha_ct|alpha_pet)$")


def _extract_state_dict(ckpt: dict) -> dict:
    """Return the network state_dict from a saved LesionLocator checkpoint."""
    sd = ckpt.get("network_weights", ckpt) if isinstance(ckpt, dict) else ckpt
    # torch.compile wraps params under a '_orig_mod.' prefix — strip it.
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def collect_weighted_levels(state_dict: dict) -> dict:
    """
    Map level index -> {'alpha_ct': tensor, 'alpha_pet': tensor} for every
    WeightedSkipFusion level found in the state dict.
    """
    levels = defaultdict(dict)
    for key, val in state_dict.items():
        m = _ALPHA_RE.search(key)
        if m:
            levels[int(m.group(1))][m.group(2)] = val.detach().float().cpu()
    return dict(levels)


def compute_split(alpha_ct: torch.Tensor, alpha_pet: torch.Tensor, eps: float = 1e-8):
    """Per-channel normalized weights, returned as (w_ct_mean, w_ct_std, n_channels)."""
    a = alpha_ct.abs()
    b = alpha_pet.abs()
    w_ct = a / (a + b + eps)
    return w_ct.mean().item(), w_ct.std().item(), w_ct.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Path to a trained checkpoint (.pth)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    levels = collect_weighted_levels(state_dict)

    if not levels:
        raise SystemExit(
            "No WeightedSkipFusion weights (fusion_modules.*.alpha_ct) found in "
            f"{args.ckpt}. Is this an intermediate-fusion checkpoint?"
        )

    # Highest level index present tells us the total number of encoder levels.
    n_levels = max(levels) + 1

    print(f"Fusion weight readout: {args.ckpt}")
    print(f"{'level':>5}  {'fusion':>16}  {'w_ct':>6}  {'w_pet':>6}  {'±std(w_ct)':>10}  {'C':>4}")
    print("-" * 60)

    ct_means = []
    for k in range(n_levels):
        if k in levels and "alpha_ct" in levels[k] and "alpha_pet" in levels[k]:
            w_ct, w_ct_std, n_ch = compute_split(levels[k]["alpha_ct"], levels[k]["alpha_pet"])
            ct_means.append(w_ct)
            print(f"{k:>5}  {'weighted':>16}  {w_ct:>6.3f}  {1 - w_ct:>6.3f}  {w_ct_std:>10.3f}  {n_ch:>4}")
        else:
            print(f"{k:>5}  {'mcsa (attention)':>16}  {'—':>6}  {'—':>6}  {'—':>10}  {'—':>4}")

    if ct_means:
        overall = sum(ct_means) / len(ct_means)
        print("-" * 60)
        print(f"mean over {len(ct_means)} weighted level(s): "
              f"w_ct={overall:.3f}  w_pet={1 - overall:.3f}")


if __name__ == "__main__":
    main()
