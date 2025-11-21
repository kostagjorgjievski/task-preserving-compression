# losses/tpc_loss.py
import torch
import torch.nn.functional as F

from .acf import acf_loss, compute_acf_batch
from .freq import freq_loss
from .periodicity import periodicity_loss
from .motif import motif_loss  # ← NEW


class TPCLoss:
    def __init__(self, cfg):
        lcfg = cfg["loss"]

        self.use_mse = lcfg["use_mse"]
        self.use_acf = lcfg["use_acf"]
        self.use_periodicity = lcfg["use_periodicity"]
        self.use_motif = lcfg["use_motif"]
        self.use_freq = lcfg["use_freq"]

        self.lambda_mse = float(lcfg["lambda_mse"])
        self.lambda_acf = float(lcfg["lambda_acf"])
        self.lambda_periodicity = float(lcfg["lambda_periodicity"])
        self.lambda_motif = float(lcfg["lambda_motif"])
        self.lambda_freq = float(lcfg["lambda_freq"])

        self.acf_cfg = lcfg.get("acf", {})
        self.freq_cfg = lcfg.get("freq", {})
        self.periodicity_cfg = lcfg.get("periodicity", {})
        self.motif_cfg = lcfg.get("motif", {})  # ← NEW

        # for shared ACF computation
        self.acf_max_lag = int(self.acf_cfg.get("max_lag", 48))
        self.seasonal_lags = self.periodicity_cfg.get("seasonal_lags", [24])
        self.acf_fn = compute_acf_batch

    def __call__(self, x, x_rec):
        """
        x, x_rec: [B, L, C]
        returns: total_loss, dict_of_components
        """
        total = torch.tensor(0.0, device=x.device)
        comps = {}

        # 1. MSE
        if self.use_mse:
            mse = F.mse_loss(x_rec, x)
            total = total + self.lambda_mse * mse
            comps["mse"] = mse.detach()

        # 2. ACF + periodicity (share ACF computation)
        if self.use_acf or self.use_periodicity:
            acf_orig = self.acf_fn(x, self.acf_max_lag)       # [B, max_lag+1]
            acf_rec = self.acf_fn(x_rec, self.acf_max_lag)    # [B, max_lag+1]

        if self.use_acf:
            lacf = acf_loss(x, x_rec, self.acf_cfg)
            total = total + self.lambda_acf * lacf
            comps["acf"] = lacf.detach()

        if self.use_periodicity:
            per = periodicity_loss(acf_orig, acf_rec, self.seasonal_lags)
            total = total + self.lambda_periodicity * per
            comps["periodicity"] = per.detach()

        # 3. Frequency
        if self.use_freq:
            lfreq = freq_loss(x, x_rec, self.freq_cfg)
            total = total + self.lambda_freq * lfreq
            comps["freq"] = lfreq.detach()

        # 4. Motif
        if self.use_motif:
            lmotif = motif_loss(x, x_rec, self.motif_cfg)
            total = total + self.lambda_motif * lmotif
            comps["motif"] = lmotif.detach()

        return total, comps
