# losses/periodicity.py
import torch

def periodicity_loss(acf_orig, acf_rec, seasonal_lags):
    """
    acf_orig, acf_rec: [B, max_lag+1]
    seasonal_lags: list of ints, e.g. [24, 48]
    """
    losses = []
    for p in seasonal_lags:
        if p < acf_orig.size(1):
            diff = acf_orig[:, p] - acf_rec[:, p]
            losses.append(diff.pow(2))
    if not losses:
        return acf_orig.new_tensor(0.0)
    return torch.mean(torch.stack(losses, dim=0))
