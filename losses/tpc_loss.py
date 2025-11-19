import torch
import torch.nn.functional as F

from .acf import acf_loss
from .freq import freq_loss


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

        # periodicity and motif will be added later and will use acf_cfg and their own config

    def __call__(self, x, x_rec):
        """
        x, x_rec: [B, L, C]
        returns: total_loss, dict_of_components
        """
        total = torch.tensor(0.0, device=x.device)
        comps = {}

        if self.use_mse:
            mse = F.mse_loss(x_rec, x)
            total = total + self.lambda_mse * mse
            comps["mse"] = mse.detach()

        if self.use_acf:
            lacf = acf_loss(x, x_rec, self.acf_cfg)
            total = total + self.lambda_acf * lacf
            comps["acf"] = lacf.detach()

        if self.use_freq:
            lfreq = freq_loss(x, x_rec, self.freq_cfg)
            total = total + self.lambda_freq * lfreq
            comps["freq"] = lfreq.detach()

        # periodicity and motif to be plugged in later

        return total, comps
