# train_tpc.py
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from utils.config import load_config
from data.ett import ETTWindowDataset
from models.registry import build_model
from losses.tpc_loss import TPCLoss

from eval.eval_structural import compute_structural_metrics_from_model


def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducible runs."""
    print(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(cfg):
    requested = cfg["training"].get("device", "auto")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device(requested)


def train(config_path: str, seed_override: int | None = None):
    cfg = load_config(config_path)

    # 0. Seed
    cfg_seed = cfg.get("training", {}).get("seed", 42)
    seed = seed_override if seed_override is not None else cfg_seed
    set_seed(seed)

    device = pick_device(cfg)
    print(f"Using device: {device}")

    # 1. Dataset and loader
    dcfg = cfg["dataset"]
    dataset = ETTWindowDataset(
        csv_path=dcfg["csv_path"],
        context_length=dcfg["context_length"],
        stride=dcfg.get("stride", 1),
    )

    num_channels = dataset.num_channels
    print(f"Dataset windows: {len(dataset)}, channels: {num_channels}")

    # Compression ratio logging
    context_len = dcfg["context_length"]
    latent_dim = cfg["model"]["latent_dim"]
    orig_dim = context_len * num_channels
    compression_ratio = orig_dim / latent_dim
    print(
        f"Compression ratio: original={orig_dim} dims -> latent={latent_dim} dims "
        f"({compression_ratio:.2f}:1)"
    )

    # Train / validation split
    vcfg = cfg.get("validation", {})
    val_frac = vcfg.get("val_frac", 0.1)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    # Use the same seed for the split generator
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=split_generator,
    )
    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    # DataLoader settings
    tcfg = cfg["training"]
    num_workers = int(tcfg.get("num_workers", 4))
    loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    print(f"DataLoader num_workers={num_workers}")

    # 2. Model
    model = build_model(cfg, num_channels=num_channels).to(device)

    # 3. Loss and optimizer
    tpc_loss = TPCLoss(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )

    max_epochs = tcfg["max_epochs"]
    log_every = cfg["logging"].get("log_every_n_steps", 50)

    ckpt_dir = Path(cfg["logging"].get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Best tracking
    best_train_loss = float("inf")
    best_struct_score = float("inf")

    best_train_path = ckpt_dir / "best_train.pt"
    best_struct_path = ckpt_dir / "best_struct.pt"

    # Structural val weighting
    alpha_acf = float(vcfg.get("alpha_acf", 1.0))
    alpha_spec = float(vcfg.get("alpha_spec", 1.0))
    val_windows = int(vcfg.get("windows", 50))
    max_lag = int(vcfg.get("max_lag", 48))
    fs = float(vcfg.get("fs", 1.0))

    # 4. Training loop
    global_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_idx, x in enumerate(loader):
            x = x.to(device)  # [B, L, C]

            opt.zero_grad()
            x_rec, z = model(x)

            loss, comps = tpc_loss(x, x_rec)
            loss.backward()
            opt.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss_sum += loss.item()

            if global_step % log_every == 0:
                msg = f"Epoch {epoch} Step {global_step} | total={loss.item():.4f}"
                for k, v in comps.items():
                    msg += f" {k}={v.item():.4f}"
                print(msg)

        # Mean train loss this epoch
        mean_train_loss = epoch_loss_sum / max(1, epoch_steps)
        print(f"Epoch {epoch} mean train loss: {mean_train_loss:.6f}")

        # Save per-epoch checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        # Update best_train.pt
        if mean_train_loss < best_train_loss:
            best_train_loss = mean_train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": cfg,
                    "mean_train_loss": mean_train_loss,
                },
                best_train_path,
            )
            print(
                f"New best_train.pt at epoch {epoch} "
                f"(mean train loss={mean_train_loss:.6f})"
            )

        # === Validation structural evaluation ===
        val_metrics = compute_structural_metrics_from_model(
            model,
            val_dataset,
            device,
            num_windows=val_windows,
            max_lag=max_lag,
            fs=fs,
        )
        val_score = (
            val_metrics["mse"]
            + alpha_acf * val_metrics["acf_dev"]
            + alpha_spec * val_metrics["spec_dev"]
        )
        print(
            f"Epoch {epoch} val structural: "
            f"mse={val_metrics['mse']:.6f} "
            f"acf_dev={val_metrics['acf_dev']:.6f} "
            f"spec_dev={val_metrics['spec_dev']:.6f} "
            f"| score={val_score:.6f}"
        )

        if val_score < best_struct_score:
            best_struct_score = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "val_score": val_score,
                },
                best_struct_path,
            )
            print(
                f"New best_struct.pt at epoch {epoch} "
                f"(score={val_score:.6f})"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/tpc_ett_tpc_minimal.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to override config training.seed",
    )
    args = parser.parse_args()

    train(args.config, seed_override=args.seed)
