import os
import subprocess
import numpy as np

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
NUM_RUNS = 5  # how many times to retrain each model

MODELS = {
    "mse_only": {
        "config": "config/tpc_ett_mse_only_rerun.yaml",
        "ckpt_dir": "checkpoints/ett_mse_only_rerun",
    },
    "tpc_best": {
        "config": "config/tpc_ett_tpc_sweep_best.yaml",
        "ckpt_dir": "checkpoints/ett_tpc_sweep_best_rerun",
    },
}

RESULTS = {
    "mse_only": {"mse": [], "acf_dev": [], "spec_dev": []},
    "tpc_best": {"mse": [], "acf_dev": [], "spec_dev": []},
}

# -------------------------------------------------------------------------
# Helper to run shell commands
# -------------------------------------------------------------------------
def run_cmd(cmd):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

# -------------------------------------------------------------------------
# Extract metrics from eval_structural output
# -------------------------------------------------------------------------
def parse_metrics(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
        line = line.strip()
        if line.startswith("mse:"):
            metrics["mse"] = float(line.split(":")[1].strip())
        elif line.startswith("acf_dev:"):
            metrics["acf_dev"] = float(line.split(":")[1].strip())
        elif line.startswith("spec_dev:"):
            metrics["spec_dev"] = float(line.split(":")[1].strip())

    required = {"mse", "acf_dev", "spec_dev"}
    if set(metrics.keys()) != required:
        raise ValueError(f"Could not parse all metrics from {output_file}: {metrics}")

    return metrics

# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
for model_key, cfg in MODELS.items():
    config_file = cfg["config"]
    ckpt_dir = cfg["ckpt_dir"]

    print("\n==============================")
    print(f" Running model: {model_key}")
    print("==============================")

    for run_idx in range(NUM_RUNS):
        print(f"\n---- Run {run_idx + 1}/{NUM_RUNS} ----")

        # 1. TRAIN MODEL
        # No --seed because train_tpc does not accept it
        run_cmd(f"python -m train_tpc --config {config_file}")

        # expected checkpoint path from YAML logging.ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, "best_struct.pt")
        if not os.path.isfile(ckpt_path):
            # fallback: search inside ckpt_dir if trainer nests folders
            print(f"[WARN] {ckpt_path} not found, searching in {ckpt_dir}...")
            found = None
            for root, _, files in os.walk(ckpt_dir):
                for f in files:
                    if f == "best_struct.pt":
                        found = os.path.join(root, f)
                        break
                if found:
                    break
            if found is None:
                raise RuntimeError(
                    f"Could not find best_struct.pt under {ckpt_dir} after training."
                )
            ckpt_path = found

        print(f"[INFO] Using checkpoint: {ckpt_path}")

        # 2. RUN EVALUATION (500 windows for stable stats)
        out_file = f"eval_{model_key}_run{run_idx}.txt"
        run_cmd(
            "python -m eval.eval_structural "
            f"--ckpt {ckpt_path} "
            f"--config {config_file} "
            f"--device cuda "
            f"--windows 500 > {out_file}"
        )

        # 3. PARSE RESULTS
        metrics = parse_metrics(out_file)
        print(f"[RESULT] Run {run_idx + 1} → {metrics}")

        RESULTS[model_key]["mse"].append(metrics["mse"])
        RESULTS[model_key]["acf_dev"].append(metrics["acf_dev"])
        RESULTS[model_key]["spec_dev"].append(metrics["spec_dev"])

# -------------------------------------------------------------------------
# SUMMARY PRINT
# -------------------------------------------------------------------------
print("\n\n==============================")
print(" FINAL MULTI-RUN RESULTS")
print("==============================")

def mean_std(x):
    x = np.array(x, dtype=float)
    return float(x.mean()), float(x.std())

for model_key in RESULTS:
    print(f"\nModel: {model_key}")

    for metric in ["mse", "acf_dev", "spec_dev"]:
        vals = RESULTS[model_key][metric]
        m, s = mean_std(vals)
        print(f"  {metric}: {m:.6f} ± {s:.6f}   (from {len(vals)} runs)")
