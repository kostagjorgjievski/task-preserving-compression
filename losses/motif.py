# losses/motif.py
import torch
import torch.nn.functional as F


def _extract_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: [B, L, C]
    returns: [B, num_patches, patch_len, C]
    """
    B, L, C = x.shape
    if L < patch_len:
        raise ValueError(f"patch_len={patch_len} larger than sequence length L={L}")

    # unfold over time dimension (dim=1)
    patches = x.unfold(dimension=1, size=patch_len, step=stride)  # [B, num_patches, patch_len, C]
    return patches


def _z_normalize_patches(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    patches: [B, num_patches, patch_len, C]
    z normalizes each patch per channel over the time axis.
    """
    # mean / std over time axis (dim=2)
    mean = patches.mean(dim=2, keepdim=True)
    std = patches.std(dim=2, keepdim=True) + eps
    return (patches - mean) / std


def motif_loss(x: torch.Tensor, x_rec: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Motif loss: encourages local shape similarity between original and reconstructed
    time series by matching z normalized patches.

    x, x_rec: [B, L, C]

    cfg options:
      patch_len: int, default 24
      stride: int, default 8
      normalize: bool, default True
      reduction: "mean" or "sum", default "mean"
    """
    patch_len = int(cfg.get("patch_len", 24))
    stride = int(cfg.get("stride", 8))
    normalize = bool(cfg.get("normalize", True))
    reduction = cfg.get("reduction", "mean")

    # extract patches
    patches_x = _extract_patches(x, patch_len=patch_len, stride=stride)
    patches_rec = _extract_patches(x_rec, patch_len=patch_len, stride=stride)

    if normalize:
        patches_x = _z_normalize_patches(patches_x)
        patches_rec = _z_normalize_patches(patches_rec)

    # both [B, num_patches, patch_len, C]
    loss = F.mse_loss(patches_rec, patches_x, reduction="none")  # [B, num_patches, patch_len, C]
    # average over time and channels
    loss = loss.mean(dim=(2, 3))  # [B, num_patches]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        # no reduction: return per batch, per patch (rarely needed)
        return loss
