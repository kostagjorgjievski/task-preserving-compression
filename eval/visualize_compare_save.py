# visualize_compare_save.py

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")   # allow saving without display

from pathlib import Path
from data.ett import ETTWindowDataset
from models.registry import build_model
from eval.structural import acf_deviation, spectrum_deviation


def load_model_and_dataset(ckpt_path):
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


def pick_window(dataset, idx):
    x = dataset[idx]      # numpy [L, C]
    x = torch.from_numpy(x).unsqueeze(0)  # [1, L, C]
    return x


def denormalize(x, dataset):
    x_np = x.squeeze(0).detach().cpu().numpy()
    mean = dataset.mean
    std = dataset.std
    return x_np * std + mean


def main(ckpt_hero, ckpt_mse, idx, channel, max_lag=48):
    cfg_h, model_h, dataset = load_model_and_dataset(ckpt_hero)
    cfg_m, model_m, dataset2 = load_model_and_dataset(ckpt_mse)

    x = pick_window(dataset, idx)

    with torch.no_grad():
        rec_h, _ = model_h(x)
        rec_m, _ = model_m(x)

    orig = denormalize(x, dataset)[:, channel]
    rec_h = denormalize(rec_h, dataset)[:, channel]
    rec_m = denormalize(rec_m, dataset)[:, channel]

    # structural metrics
    acf_h, acf_orig, acf_hv = acf_deviation(orig, rec_h, max_lag)
    acf_m, _, acf_mv = acf_deviation(orig, rec_m, max_lag)

    spec_h, (f_o, S_o), (f_h, S_h) = spectrum_deviation(orig, rec_h)
    spec_m, (_, _), (f_m, S_m) = spectrum_deviation(orig, rec_m)

    # Plot
    t = np.arange(len(orig))
    lags = np.arange(max_lag + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(t, orig, label="original")
    axes[0].plot(t, rec_m, label="mse-only")
    axes[0].plot(t, rec_h, label="hero")
    axes[0].set_title("Time series")
    axes[0].legend()

    axes[1].plot(lags, acf_orig, label="orig")
    axes[1].plot(lags, acf_mv, label="mse-only")
    axes[1].plot(lags, acf_hv, label="hero")
    axes[1].set_title("ACF")
    axes[1].legend()

    axes[2].plot(f_o, S_o, label="orig")
    axes[2].plot(f_m, S_m, label="mse-only")
    axes[2].plot(f_h, S_h, label="hero")
    axes[2].set_title("Spectrum")
    axes[2].legend()

    plt.tight_layout()

    out_path = f"compare_window{idx}_channel{channel}.png"
    plt.savefig(out_path, dpi=150)
    print("Saved:", out_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_hero", required=True)
    p.add_argument("--ckpt_mse", required=True)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--channel", type=int, default=0)
    args = p.parse_args()
    main(args.ckpt_hero, args.ckpt_mse, args.idx, args.channel)
