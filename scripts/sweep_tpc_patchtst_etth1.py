import csv
import subprocess
import shlex
from pathlib import Path

# Paths relative to project root
ROOT = Path(__file__).resolve().parents[1]
LAMBDA_GRID_CSV = ROOT / "config" / "tpc_lambda_grid.csv"
TPC_CONFIG = ROOT / "config" / "tpc_ett_tpc_sweep_best.yaml"
TPC_CKPT_ROOT = ROOT / "checkpoints" / "ett_tpc_lambda_sweep"
ETTH1_RAW = ROOT / "forecasting" / "patchtst" / "dataset" / "ETTh1.csv"
ETTH1_TPC_DIR = ROOT / "forecasting" / "patchtst" / "dataset"
PATCHTST_DIR = ROOT / "forecasting" / "patchtst"
LOG_DIR = PATCHTST_DIR / "logs" / "LongForecasting"
RESULT_CSV = ROOT / "results" / "tpc_patchtst_lambda_sweep.csv"

PRED_LENS = [96, 192, 336, 720]  # horizons to evaluate


def run_cmd(cmd, cwd=None, log_file=None):
    """Run a shell command. Optionally tee output to a log file."""
    print(f"\n[RUN] {cmd}")
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")
    else:
        subprocess.run(cmd, cwd=cwd, shell=True, check=True)


def train_tpc(run):
    """Train one TPC model for a given lambda config."""
    run_id = run["run_id"]
    lambda_acf = run["lambda_acf"]
    lambda_freq = run["lambda_freq"]
    lambda_motif = run["lambda_motif"]
    lambda_cp = run["lambda_cp"]
    lambda_per = run["lambda_per"]

    ckpt_dir = TPC_CKPT_ROOT / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TODO: replace this with the exact command you used before to train TPC
    # Important: make sure it:
    #  - saves best_train.pt into ckpt_dir
    #  - accepts lambda_* as CLI flags or reads them from env and logs them
    cmd = " ".join([
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONPATH=.",
        "python", "-m", "train_tpc",
        f"--config {TPC_CONFIG}",
        f"--ckpt_dir {ckpt_dir}",
        "--device cuda",
        f"--lambda_acf {lambda_acf}",
        f"--lambda_freq {lambda_freq}",
        f"--lambda_motif {lambda_motif}",
        f"--lambda_cp {lambda_cp}",
        f"--lambda_per {lambda_per}",
    ])

    log_file = ROOT / "logs" / "lambda_sweep_tpc" / f"{run_id}.log"
    run_cmd(cmd, cwd=ROOT, log_file=log_file)

    ckpt = ckpt_dir / "best_train.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt}")
    return ckpt


def build_etth1_tpc(run_id, ckpt_path):
    """Build a compressed ETTh1 CSV using a trained TPC model."""
    out_csv = ETTH1_TPC_DIR / f"ETTh1_tpc_{run_id}.csv"
    cmd = " ".join([
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONPATH=.",
        "python", "scripts/build_etth1_tpc_csv.py",
        f"--ckpt {ckpt_path}",
        f"--config {TPC_CONFIG}",
        f"--etth1_in {ETTH1_RAW}",
        f"--etth1_out {out_csv}",
        "--device cuda",
    ])
    log_file = ROOT / "logs" / "lambda_sweep_build" / f"{run_id}.log"
    run_cmd(cmd, cwd=ROOT, log_file=log_file)
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected compressed ETTh1 not found: {out_csv}")
    return out_csv


def train_patchtst_on_compressed(run_id, etth1_tpc_path):
    """Train PatchTST on compressed ETTh1 for each horizon. Return list of metric dicts."""
    metrics = []

    for pred_len in PRED_LENS:
        model_id = f"ETTh1_TPC_{run_id}_336_{pred_len}"
        log_file = LOG_DIR / f"PatchTST_{model_id}.log"

        cmd = " ".join([
            "CUDA_VISIBLE_DEVICES=0",
            "python", "run_longExp.py",
            "--model PatchTST",
            "--data ETTh1",
            "--root_path ./dataset/",
            f"--data_path {etth1_tpc_path.name}",
            "--features M",
            "--target OT",
            "--freq h",
            "--seq_len 336",
            "--label_len 48",
            f"--pred_len {pred_len}",
            "--enc_in 7",
            "--dec_in 7",
            "--c_out 7",
            "--d_model 16",
            "--n_heads 4",
            "--e_layers 3",
            "--d_layers 1",
            "--d_ff 128",
            "--moving_avg 25",
            "--factor 1",
            "--distil",
            "--dropout 0.3",
            "--fc_dropout 0.3",
            "--embed timeF",
            "--activation gelu",
            "--num_workers 10",
            "--itr 1",
            "--train_epochs 100",
            "--batch_size 32",
            "--learning_rate 0.0001",
            "--loss mse",
            "--lradj type3",
            "--pct_start 0.3",
            "--revin",
            "--use_gpu",
            "--gpu 0",
            f"--des Exp_TPC_{run_id}",
            f"--model_id {model_id}",
        ])

        run_cmd(cmd, cwd=PATCHTST_DIR, log_file=log_file)
        m = parse_patchtst_log(log_file, run_id, pred_len)
        metrics.append(m)

    return metrics


def parse_patchtst_log(log_path, run_id, pred_len):
    """Parse the last 'mse:..., mae:..., rse:...' line from a PatchTST log."""
    log_path = Path(log_path)
    mse = mae = rse = None

    with log_path.open() as f:
        for line in f:
            if "mse:" in line and "mae:" in line and "rse:" in line:
                # this will keep overwriting, so at the end we keep the last metrics
                parts = line.strip().split(",")
                kv = {}
                for p in parts:
                    if ":" in p:
                        k, v = p.split(":", 1)
                        kv[k.strip()] = float(v.strip())
                mse = kv.get("mse", mse)
                mae = kv.get("mae", mae)
                rse = kv.get("rse", rse)

    if mse is None:
        raise ValueError(f"Could not find mse line in log {log_path}")

    return {
        "run_id": run_id,
        "pred_len": pred_len,
        "mse": mse,
        "mae": mae,
        "rse": rse,
    }


def load_lambda_grid():
    runs = []
    with LAMBDA_GRID_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append({
                "run_id": row["run_id"],
                "lambda_acf": float(row["lambda_acf"]),
                "lambda_freq": float(row["lambda_freq"]),
                "lambda_motif": float(row["lambda_motif"]),
                "lambda_cp": float(row["lambda_cp"]),
                "lambda_per": float(row["lambda_per"]),
            })
    return runs


def append_results(rows):
    """Append rows (list of dicts) to RESULT_CSV, creating header if needed."""
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_id",
        "lambda_acf",
        "lambda_freq",
        "lambda_motif",
        "lambda_cp",
        "lambda_per",
        "pred_len",
        "mse",
        "mae",
        "rse",
    ]
    file_exists = RESULT_CSV.exists()
    with RESULT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    runs = load_lambda_grid()
    print(f"Loaded {len(runs)} lambda configs")

    for i, run in enumerate(runs, start=1):
        run_id = run["run_id"]
        print(f"\n===== [{i}/{len(runs)}] run_id = {run_id} =====")

        # 1. train TPC
        ckpt = train_tpc(run)

        # 2. build compressed ETTh1 for this run
        etth1_tpc = build_etth1_tpc(run_id, ckpt)

        # 3. train PatchTST on compressed data for all horizons
        metrics = train_patchtst_on_compressed(run_id, etth1_tpc)

        # 4. append results
        rows = []
        for m in metrics:
            rows.append({
                **run,
                "pred_len": m["pred_len"],
                "mse": m["mse"],
                "mae": m["mae"],
                "rse": m["rse"],
            })
        append_results(rows)


if __name__ == "__main__":
    main()
