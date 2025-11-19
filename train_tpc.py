import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from data.ett import ETTWindowDataset
from models.registry import build_model
from losses.tpc_loss import TPCLoss

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




def train(config_path):
    cfg = load_config(config_path)

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

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # 2. Model
    model = build_model(cfg, num_channels=num_channels).to(device)

    # 3. Loss and optimizer
    tpc_loss = TPCLoss(cfg["loss"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    max_epochs = cfg["training"]["max_epochs"]
    log_every = cfg["logging"].get("log_every_n_steps", 50)

    # 4. Training loop
    global_step = 0
    model.train()

    ckpt_dir = Path(cfg["logging"].get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        for batch_idx, x in enumerate(loader):
            x = x.to(device)  # [B, L, C]

            opt.zero_grad()
            x_rec, z = model(x)

            loss, comps = tpc_loss(x, x_rec)
            loss.backward()
            opt.step()

            global_step += 1

            if global_step % log_every == 0:
                msg = f"Epoch {epoch} Step {global_step} | total={loss.item():.4f}"
                for k, v in comps.items():
                    msg += f" {k}={v.item():.4f}"
                print(msg)

        # save a checkpoint every epoch
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/tpc_ett_tpc_minimal.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    train(args.config)
