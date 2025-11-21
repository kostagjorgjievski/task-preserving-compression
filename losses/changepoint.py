# losses/changepoint.py
import torch
import torch.nn.functional as F


def _first_diff(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, C]
    returns: [B, L-1, C] differences along time
    """
    return x[:, 1:, :] - x[:, :-1, :]


def changepoint_loss(x: torch.Tensor, x_rec: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Changepoint loss: encourage original and reconstructed series to have
    similar jump structure (changepoints) along time.

    x, x_rec: [B, L, C]

    cfg options:
      use_abs: bool, default True
      use_tanh: bool, default True
      scale: float, default 5.0      (for tanh scaling)
      reduction: "mean" or "sum", default "mean"
    """
    use_abs = bool(cfg.get("use_abs", True))
    use_tanh = bool(cfg.get("use_tanh", True))
    scale = float(cfg.get("scale", 5.0))
    reduction = cfg.get("reduction", "mean")

    d_x = _first_diff(x)       # [B, L-1, C]
    d_rec = _first_diff(x_rec)

    if use_abs:
        d_x = d_x.abs()
        d_rec = d_rec.abs()

    if use_tanh:
        # squash very large jumps but keep ordering
        d_x = torch.tanh(scale * d_x)
        d_rec = torch.tanh(scale * d_rec)

    loss = F.mse_loss(d_rec, d_x, reduction="none")  # [B, L-1, C]
    loss = loss.mean(dim=(1, 2))  # [B]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        # no global reduction
        return loss
