# visualize_recon.py
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
    model.load_state_dict(ckpt["model_state_dict"])
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


def main(ckpt_path, idx=0, channel=0, max_lag=48):
    cfg, model, dataset = load_model_and_data(ckpt_path)

    # pick one window
    x = pick_window(dataset, idx=idx)      # [1, L, C]
    with torch.no_grad():
        x_rec, z = model(x)

    # denormalize
    x_den = denormalize(x, dataset)        # [L, C]
    x_rec_den = denormalize(x_rec, dataset)

    # pick channel
    orig = x_den[:, channel]
    rec = x_rec_den[:, channel]

    # structural metrics
    acf_dev, acf_orig, acf_rec = acf_deviation(orig, rec, max_lag=max_lag)
    spec_dev, (f1, S1), (f2, S2) = spectrum_deviation(orig, rec, fs=1.0)

    print(f"ACF deviation: {acf_dev:.6f}")
    print(f"Spectrum deviation: {spec_dev:.6f}")

    # plotting
    L = len(orig)
    t = np.arange(L)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1 - time series
    axes[0].plot(t, orig, label="original")
    axes[0].plot(t, rec, label="reconstructed", alpha=0.7)
    axes[0].set_title(f"Time series - channel {channel}, window {idx}")
    axes[0].legend()
    axes[0].grid(True)

    # 2 - ACF
    lags = np.arange(max_lag + 1)
    axes[1].plot(lags, acf_orig, marker="o", label="original ACF")
    axes[1].plot(lags, acf_rec, marker="x", label="reconstructed ACF")
    axes[1].set_title("ACF comparison")
    axes[1].legend()
    axes[1].grid(True)

    # 3 - Spectrum
    axes[2].plot(f1, S1, label="original spectrum")
    axes[2].plot(f2, S2, label="reconstructed spectrum", alpha=0.7)
    axes[2].set_title("Spectrum comparison (Welch)")
    axes[2].set_xlabel("Frequency")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/ett_tpc_minimal/epoch_30.pt",
        help="Path to checkpoint file",
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
    args = parser.parse_args()

    main(args.ckpt, idx=args.idx, channel=args.channel)
