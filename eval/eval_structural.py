# eval/eval_structural.py

import argparse
import numpy as np
import torch

from utils.config import load_config
from data.ett import ETTWindowDataset
from models.registry import build_model
from eval.structural import acf_deviation, spectrum_deviation


def compute_structural_metrics_from_model(
    model,
    dataset,
    device,
    num_windows=50,
    max_lag=48,
    fs=1.0,
):
    model.eval()

    n = len(dataset)
    k = min(num_windows, n)
    idxs = np.random.choice(n, size=k, replace=False)

    mse_list = []
    acf_list = []
    spec_list = []

    with torch.no_grad():
        for idx in idxs:
            x = dataset[idx]  # could be numpy array or tensor

            # Normalize to a torch tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            # if your dataset ever returns (x, y), handle that too:
            # if isinstance(x, (tuple, list)):
            #     x = x[0]

            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, L, C]

            x = x.to(device)

            x_rec, _ = model(x)

            # move to CPU numpy, use first channel for structural metrics
            x_np = x.cpu().numpy()[0, :, 0]
            x_rec_np = x_rec.cpu().numpy()[0, :, 0]

            mse = np.mean((x_np - x_rec_np) ** 2)
            mse_list.append(mse)

            acf_dev, _, _ = acf_deviation(x_np, x_rec_np, max_lag=max_lag)
            spec_dev, _, _ = spectrum_deviation(x_np, x_rec_np, fs=fs)

            acf_list.append(acf_dev)
            spec_list.append(spec_dev)

    model.train()

    return {
        "mse": float(np.mean(mse_list)),
        "acf_dev": float(np.mean(acf_list)),
        "spec_dev": float(np.mean(spec_list)),
    }



def evaluate_structural(
    ckpt_path,
    config_path,
    device_str="cpu",
    num_windows=50,
    max_lag=48,
    fs=1.0,
):
    """
    CLI helper: load model from checkpoint and compute structural metrics.
    """
    device = torch.device(device_str)

    # Load config
    cfg = load_config(config_path)

    # Dataset
    dcfg = cfg["dataset"]
    dataset = ETTWindowDataset(
        csv_path=dcfg["csv_path"],
        context_length=dcfg["context_length"],
        stride=dcfg.get("stride", 1),
    )

    num_channels = dataset.num_channels

    # Model
    model = build_model(cfg, num_channels=num_channels).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Compute structural metrics
    metrics = compute_structural_metrics_from_model(
        model,
        dataset,
        device,
        num_windows=num_windows,
        max_lag=max_lag,
        fs=fs,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config used for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation (cpu / cuda / mps)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=50,
        help="Number of random windows to evaluate",
    )
    parser.add_argument(
        "--max_lag",
        type=int,
        default=48,
        help="Max lag for ACF",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1.0,
        help="Sampling frequency for spectrum",
    )

    args = parser.parse_args()

    metrics = evaluate_structural(
        ckpt_path=args.ckpt,
        config_path=args.config,
        device_str=args.device,
        num_windows=args.windows,
        max_lag=args.max_lag,
        fs=args.fs,
    )

    print("\n=== Structural Evaluation ===")
    print(f"mse: {metrics['mse']:.6f}")
    print(f"acf_dev: {metrics['acf_dev']:.6f}")
    print(f"spec_dev: {metrics['spec_dev']:.6f}")
