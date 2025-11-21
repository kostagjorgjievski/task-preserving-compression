# losses/acf.py
import torch


def compute_acf_batch(x, max_lag: int):
    """
    x: [B, L, C]
    returns: acf [B, max_lag+1], averaged over channels.
    """
    B, L, C = x.shape

    # [B, L, C] -> [B, C, L] -> [B*C, L]
    x_bc = x.permute(0, 2, 1).reshape(B * C, L)

    # center per (batch, channel)
    x_bc = x_bc - x_bc.mean(dim=1, keepdim=True)

    # denominator is variance * L (sum of squares)
    denom = (x_bc ** 2).sum(dim=1) + 1e-8  # [B*C]

    acf = x_bc.new_empty(B * C, max_lag + 1)

    for k in range(max_lag + 1):
        # overlap region length L - k
        prod = x_bc[:, : L - k] * x_bc[:, k:]
        num = prod.sum(dim=1)          # [B*C]
        acf[:, k] = num / denom        # normalized autocorr

    # [B*C, max_lag+1] -> [B, C, max_lag+1] -> average over C
    acf = acf.view(B, C, max_lag + 1).mean(dim=1)   # [B, max_lag+1]
    return acf


def acf_loss(x, x_rec, cfg):
    """
    Scalar ACF MSE loss used in training.
    """
    max_lag = int(cfg.get("max_lag", 48))
    acf_orig = compute_acf_batch(x, max_lag)
    acf_rec = compute_acf_batch(x_rec, max_lag)
    return torch.mean((acf_orig - acf_rec) ** 2)
