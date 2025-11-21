# visualize_recon_pair.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.config import load_config
from data.ett import ETTWindowDataset
from models.registry import build_model
from eval.structural import acf_deviation, spectrum_deviation


def load_model_and_data(ckpt_path):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    dcfg = cfg["dataset"]
    dataset = ETTWindowDataset(
        csv_path=dcfg["csv_path"],
        context_length=dcfg["context_length"],
        stride=dcfg.get("stride", 1),
    )

    num_channels = dataset.num_channels
    model = build_model(cfg, num_channels=num_channels)
    # adapt key name if needed ("model_state_dict" vs "model_state")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()

    return cfg, model, dataset


def pick_window(dataset, idx=0):
    # returns normalized window [L, C] as torch tensor [1, L, C]
    x = dataset[idx]  # [L, C] numpy
    x = torch.from_numpy(x).unsqueeze(0)  # [1, L, C]
    return x


def denormalize(x, dataset):
    """
    x: [1, L, C] torch
    dataset.mean, dataset.std: [1, C] numpy
    returns: [L, C] numpy in original scale
    """
    x_np = x.squeeze(0).detach().cpu().numpy()  # [L, C]
    mean = dataset.mean  # [1, C]
    std = dataset.std    # [1, C]
    x_denorm = x_np * std + mean
    return x_denorm


def main(ckpt_mse, ckpt_tpc, idx=0, channel=0, max_lag=48):
    # load MSE only
    cfg_mse, model_mse, dataset = load_model_and_data(ckpt_mse)
    # load TPC model, reuse same dataset object
    _, model_tpc, _ = load_model_and_data(ckpt_tpc)

    # pick one window from dataset
    x = pick_window(dataset, idx=idx)      # [1, L, C]

    with torch.no_grad():
        x_rec_mse, z_mse = model_mse(x)
        x_rec_tpc, z_tpc = model_tpc(x)

    # denormalize all
    x_den = denormalize(x, dataset)            # [L, C]
    x_rec_mse_den = denormalize(x_rec_mse, dataset)
    x_rec_tpc_den = denormalize(x_rec_tpc, dataset)

    # pick channel
    orig = x_den[:, channel]
    rec_mse = x_rec_mse_den[:, channel]
    rec_tpc = x_rec_tpc_den[:, channel]

    # structural metrics for both
    acf_dev_mse, acf_orig, acf_mse = acf_deviation(orig, rec_mse, max_lag=max_lag)
    acf_dev_tpc, _, acf_tpc = acf_deviation(orig, rec_tpc, max_lag=max_lag)

    spec_dev_mse, (f1, S1), (f_mse, S_mse) = spectrum_deviation(orig, rec_mse, fs=1.0)
    spec_dev_tpc, (_, _), (f_tpc, S_tpc) = spectrum_deviation(orig, rec_tpc, fs=1.0)

    print(f"[Window {idx}, channel {channel}]")
    print(f"MSE only   - ACF deviation: {acf_dev_mse:.6f}, Spectrum deviation: {spec_dev_mse:.6f}")
    print(f"BalancedTP - ACF deviation: {acf_dev_tpc:.6f}, Spectrum deviation: {spec_dev_tpc:.6f}")

    # plotting
    L = len(orig)
    t = np.arange(L)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1 - time series
    axes[0].plot(t, orig, label="original")
    axes[0].plot(t, rec_mse, label="MSE-only recon", alpha=0.7)
    axes[0].plot(t, rec_tpc, label="Balanced TPC recon", alpha=0.7)
    axes[0].set_title(f"Time series - channel {channel}, window {idx}")
    axes[0].legend()
    axes[0].grid(True)

    # 2 - ACF
    lags = np.arange(max_lag + 1)
    axes[1].plot(lags, acf_orig, marker="o", label="original ACF")
    axes[1].plot(lags, acf_mse, marker="x", label="MSE-only ACF")
    axes[1].plot(lags, acf_tpc, marker="s", label="Balanced TPC ACF")
    axes[1].set_title("ACF comparison")
    axes[1].legend()
    axes[1].grid(True)

    # 3 - Spectrum (Welch from your spectrum_deviation)
    axes[2].plot(f1, S1, label="original spectrum")
    axes[2].plot(f_mse, S_mse, label="MSE-only spectrum", alpha=0.7)
    axes[2].plot(f_tpc, S_tpc, label="Balanced TPC spectrum", alpha=0.7)
    axes[2].set_title("Spectrum comparison (Welch)")
    axes[2].set_xlabel("Frequency")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    # if you prefer saving:
    # plt.savefig(f"plots/window_{idx}_ch_{channel}_mse_vs_tpc.png")
    # plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_mse",
        type=str,
        default="checkpoints/ett_mse_only/best_struct.pt",
        help="Path to MSE-only checkpoint file",
    )
    parser.add_argument(
        "--ckpt_tpc",
        type=str,
        default="checkpoints/ett_tpc_balanced/best_struct.pt",
        help="Path to Balanced TPC checkpoint file",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Dataset window index to visualize",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to visualize",
    )
    parser.add_argument(
        "--max_lag",
        type=int,
        default=48,
        help="Maximum lag for ACF",
    )
    args = parser.parse_args()

    main(
        ckpt_mse=args.ckpt_mse,
        ckpt_tpc=args.ckpt_tpc,
        idx=args.idx,
        channel=args.channel,
        max_lag=args.max_lag,
    )
