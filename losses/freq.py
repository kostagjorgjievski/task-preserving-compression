# tpc/losses/freq.py
import torch


def freq_loss(x, x_rec, freq_cfg):
    """
    x, x_rec: [B, L, C]
    freq_cfg:
        - use_full_spectrum: bool
        - max_bins: int or None
        - use_log: bool (new)
        - normalize: bool (new)
    """
    use_full = bool(freq_cfg.get("use_full_spectrum", True))
    max_bins = freq_cfg.get("max_bins", None)
    use_log = bool(freq_cfg.get("use_log", True))
    normalize = bool(freq_cfg.get("normalize", True))

    B, L, C = x.shape

    # [B, C, L]
    x_c = x.permute(0, 2, 1)
    xr_c = x_rec.permute(0, 2, 1)

    # remove mean to ignore DC bias
    x_c = x_c - x_c.mean(dim=2, keepdim=True)
    xr_c = xr_c - xr_c.mean(dim=2, keepdim=True)

    # rFFT
    X = torch.fft.rfft(x_c, dim=2)
    Xr = torch.fft.rfft(xr_c, dim=2)

    mag_X = torch.abs(X)      # [B, C, F]
    mag_Xr = torch.abs(Xr)

    if not use_full and max_bins is not None:
        max_bins = int(max_bins)
        mag_X = mag_X[:, :, :max_bins]
        mag_Xr = mag_Xr[:, :, :max_bins]

    eps = 1e-8

    if normalize:
        # normalize each (B, C, :) spectrum along freq axis
        mag_X = mag_X / (mag_X.sum(dim=-1, keepdim=True) + eps)
        mag_Xr = mag_Xr / (mag_Xr.sum(dim=-1, keepdim=True) + eps)

    if use_log:
        mag_X = torch.log(mag_X + eps)
        mag_Xr = torch.log(mag_Xr + eps)

    loss = torch.mean((mag_X - mag_Xr) ** 2)
    return loss
