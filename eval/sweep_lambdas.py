# sweep_lambdas.py
import itertools
import os
import subprocess

import torch
import yaml

from utils.config import load_config 

BASE_CFG = "config/tpc_ett_tpc_sweep_base.yaml"
RESULTS_CSV = "lambda_sweep_results.csv"


def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def run_one(cfg_path, ckpt_dir):
    """
    Train + evaluate one config.
    Returns dict with metrics parsed from eval_structural stdout.
    """
    print(f"\n=== Training {ckpt_dir} ===")
    env = os.environ.copy()

    # 1) Train
    subprocess.run(
        ["python", "-m", "train_tpc", "--config", cfg_path],
        check=True,
        env=env,
    )

    # 2) Evaluate best_struct
    ckpt_path = os.path.join(ckpt_dir, "best_struct.pt")
    print(f"Evaluating {ckpt_path}")

    proc = subprocess.run(
        [
            "python",
            "-m",
            "eval.eval_structural",
            "--ckpt",
            ckpt_path,
            "--config",
            cfg_path,
            "--windows",
            "50",
            "--device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    # parse the printed metrics from stdout
    # Expected lines:
    # === Structural Evaluation ===
    # mse: 0.123456
    # acf_dev: 0.012345
    # spec_dev: 4.567890
    out = proc.stdout.strip().splitlines()
    metrics = {}
    for line in out:
        line = line.strip()
        if line.startswith("mse:"):
            metrics["mse"] = float(line.split()[1])
        elif line.startswith("acf_dev:"):
            metrics["acf_dev"] = float(line.split()[1])
        elif line.startswith("spec_dev:"):
            metrics["spec_dev"] = float(line.split()[1])

    print("Metrics:", metrics)
    return metrics


def main():
    base_cfg = load_config(BASE_CFG)

    # search spaces
    lambda_acf_vals = [0.1, 0.2, 0.4, 0.8]
    lambda_freq_vals = [0.08, 0.12, 0.2]     # no 0.04 to keep search sane
    lambda_motif_vals = [0.02, 0.05, 0.1]
    lambda_cp_vals = [0.01, 0.02]            # changepoint
    lambda_per_vals = [0.01, 0.02]           # periodicity

    # prepare CSV if not present
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w") as f:
            f.write(
                "run_id,lambda_acf,lambda_freq,lambda_motif,lambda_cp,lambda_per,"
                "mse,acf_dev,spec_dev\n"
            )

    total_runs = 0
    for lam_acf, lam_freq, lam_motif, lam_cp, lam_per in itertools.product(
        lambda_acf_vals, lambda_freq_vals, lambda_motif_vals, lambda_cp_vals, lambda_per_vals
    ):
        # constraints to keep configs reasonable and total count ~144
        # 1) motif not insanely bigger than freq
        if lam_motif > 2 * lam_freq:
            continue
        # 2) changepoint penalty should not dominate ACF
        if lam_cp > lam_acf / 2:
            continue
        # 3) periodicity not larger than ACF or freq
        if lam_per > lam_acf:
            continue
        if lam_per > lam_freq:
            continue

        run_id = (
            f"acf{lam_acf}_freq{lam_freq}_mot{lam_motif}_"
            f"cp{lam_cp}_per{lam_per}"
        )
        run_id = run_id.replace(".", "p")
        ckpt_dir = os.path.join("checkpoints", f"ett_tpc_sweep_{run_id}")

        # resume: if we've already got a best_struct, skip
        if os.path.exists(os.path.join(ckpt_dir, "best_struct.pt")):
            print(f"Skipping {run_id}, checkpoint already exists.")
            continue

        # clone base config and override lambdas
        cfg = yaml.safe_load(open(BASE_CFG))

        cfg["loss"]["use_acf"] = True
        cfg["loss"]["use_freq"] = True
        cfg["loss"]["use_motif"] = True
        cfg["loss"]["use_changepoint"] = True
        cfg["loss"]["use_periodicity"] = True

        cfg["loss"]["lambda_acf"] = float(lam_acf)
        cfg["loss"]["lambda_freq"] = float(lam_freq)
        cfg["loss"]["lambda_motif"] = float(lam_motif)
        cfg["loss"]["lambda_changepoint"] = float(lam_cp)
        cfg["loss"]["lambda_periodicity"] = float(lam_per)

        # unique ckpt_dir per run
        cfg.setdefault("logging", {})
        cfg["logging"]["ckpt_dir"] = ckpt_dir

        # temp config path
        tmp_cfg_path = f"config/_tmp_sweep_{run_id}.yaml"
        save_yaml(cfg, tmp_cfg_path)

        try:
            metrics = run_one(tmp_cfg_path, ckpt_dir)
        except subprocess.CalledProcessError as e:
            print(f"Run {run_id} failed:", e)
            continue

        # append to CSV
        with open(RESULTS_CSV, "a") as f:
            f.write(
                f"{run_id},{lam_acf},{lam_freq},{lam_motif},{lam_cp},{lam_per},"
                f"{metrics.get('mse', float('nan'))},"
                f"{metrics.get('acf_dev', float('nan'))},"
                f"{metrics.get('spec_dev', float('nan'))}\n"
            )

        total_runs += 1
        print(f"Completed runs so far: {total_runs}")

    print(f"Finished sweep. Total successful runs: {total_runs}")


if __name__ == "__main__":
    main()
