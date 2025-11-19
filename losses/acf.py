import torch


def _acf_1d(x, max_lag):
    """
    x: [B, L] zero mean
    returns: [B, max_lag + 1]
    """
    B, L = x.shape
    denom = (x ** 2).sum(dim=1, keepdim=True) + 1e-8  # [B, 1]

    acfs = []
    for k in range(max_lag + 1):
        if k == 0:
            num = denom
        else:
            # x[:, :-k] * x[:, k:]
            num = (x[:, :-k] * x[:, k:]).sum(dim=1, keepdim=True)
        acfs.append(num / denom)

    acf = torch.cat(acfs, dim=1)  # [B, max_lag + 1]
    return acf


def acf_loss(x, x_rec, acf_cfg):
    """
    x, x_rec: [B, L, C]
    acf_cfg: dict with keys:
        - max_lag: int
        - periodic_lags: list[int] (not used here, but can support periodic-only variant)
    We compute full ACF up to max_lag for each channel, then L2 difference.
    """
    max_lag = int(acf_cfg.get("max_lag", 48))

    # center along time
    x_c = x - x.mean(dim=1, keepdim=True)          # [B, L, C]
    xr_c = x_rec - x_rec.mean(dim=1, keepdim=True)

    B, L, C = x.shape

    # reshape to 2D for acf: [B*C, L]
    x_flat = x_c.permute(0, 2, 1).contiguous().view(B * C, L)
    xr_flat = xr_c.permute(0, 2, 1).contiguous().view(B * C, L)

    acf_x = _acf_1d(x_flat, max_lag)    # [B*C, max_lag+1]
    acf_xr = _acf_1d(xr_flat, max_lag)

    loss = torch.mean((acf_x - acf_xr) ** 2)
    return loss
