#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.config import load_config
from models.registry import build_model


def load_tpc_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, num_channels=cfg["model"]["num_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def sliding_windows(arr: np.ndarray, window: int, stride: int = 1):
    """
    arr: [T, C]
    return: [N, window, C], starts: [N]
    """
    T = arr.shape[0]
    starts = list(range(0, T - window + 1, stride))
    windows = np.stack([arr[s : s + window] for s in starts], axis=0)
    return windows, np.array(starts, dtype=np.int64)


def reconstruct_series_from_windows(windows_rec: np.ndarray,
                                    starts: np.ndarray,
                                    T: int) -> np.ndarray:
    """
    windows_rec: [N, L, C] reconstructed windows
    starts: [N] start index for each window
    T: total original length
    Returns: recon[T, C] by overlap-averaging.
    """
    N, L, C = windows_rec.shape
    sums = np.zeros((T, C), dtype=np.float32)
    counts = np.zeros((T, C), dtype=np.float32)

    for w, s in zip(windows_rec, starts):
        sums[s : s + L] += w
        counts[s : s + L] += 1.0

    counts[counts == 0] = 1.0
    return sums / counts


@torch.no_grad()
def compress_etth1_to_csv(
    tpc_ckpt: Path,
    tpc_config: Path,
    etth1_csv_in: Path,
    etth1_csv_out: Path,
    device_str: str = "cuda",
    batch_size: int = 256,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1. Load config (for context_length etc.)
    cfg = load_config(str(tpc_config))
    context_len = cfg["dataset"]["context_length"]
    stride = cfg["dataset"].get("stride", 1)
    num_channels = cfg["model"]["num_channels"]

    # 2. Load model
    print(f"Loading TPC model from {tpc_ckpt}")
    model, _ = load_tpc_model(tpc_ckpt, device)

    # 3. Load ETTh1 raw csv (PatchTST style: date + 7 vars)
    print(f"Loading ETTh1 from {etth1_csv_in}")
    df = pd.read_csv(etth1_csv_in)
    assert "date" in df.columns, "Expected a 'date' column in ETTh1.csv"

    feature_cols = [c for c in df.columns if c != "date"]
    data = df[feature_cols].values.astype(np.float32)
    T, C = data.shape
    assert C == num_channels, f"Expected {num_channels} channels, got {C}"

    # 4. Build sliding windows
    print(f"Building sliding windows L={context_len}, stride={stride}")
    windows, starts = sliding_windows(data, window=context_len, stride=stride)
    N = windows.shape[0]
    print(f"Total windows: {N}")

    # 5. Run through autoencoder and get recon windows
    rec_windows = np.zeros_like(windows, dtype=np.float32)
    idx = 0
    while idx < N:
        j = min(idx + batch_size, N)
        batch = torch.from_numpy(windows[idx:j]).to(device)  # [B, L, C]

        # model should return x_rec, z
        x_rec, _ = model(batch)
        rec_windows[idx:j] = x_rec.detach().cpu().numpy()
        idx = j
        if idx % (batch_size * 10) == 0 or idx == N:
            print(f"Processed {idx}/{N} windows")

    # 6. Overlap-average reconstruction to full series
    print("Reconstructing full series by overlap-averaging")
    recon = reconstruct_series_from_windows(rec_windows, starts, T=T)

    # 7. Write new csv with same structure but reconstructed values
    out_df = df.copy()
    out_df[feature_cols] = recon

    etth1_csv_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(etth1_csv_out, index=False)
    print(f"Saved compressed ETTh1 to {etth1_csv_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to TPC checkpoint (best_train.pt or best_struct.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TPC yaml config (e.g. config/tpc_ett_tpc_sweep_best.yaml)",
    )
    parser.add_argument(
        "--etth1_in",
        type=str,
        default="forecasting/patchtst/dataset/ETTh1.csv",
        help="Input ETTh1 csv (original)",
    )
    parser.add_argument(
        "--etth1_out",
        type=str,
        default="forecasting/patchtst/dataset/ETTh1_tpc.csv",
        help="Output ETTh1 csv (reconstructed/compressed)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    compress_etth1_to_csv(
        Path(args.ckpt),
        Path(args.config),
        Path(args.etth1_in),
        Path(args.etth1_out),
        device_str=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
