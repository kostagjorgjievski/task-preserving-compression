import csv
from pathlib import Path

# Paths relative to repo root
ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "lambda_sweep_results.csv"        # from sweep_lambdas.py
OUT_CSV = ROOT / "config" / "tpc_lambda_grid.csv" # for PatchTST sweep

TOP_K = 10  # how many best configs you want to keep


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    rows = []
    with IN_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                mse = float(r["mse"])
            except (KeyError, ValueError):
                continue
            rows.append((mse, r))

    if not rows:
        raise RuntimeError("No valid rows with mse found in lambda_sweep_results.csv")

    # sort by mse ascending
    rows.sort(key=lambda x: x[0])
    top = [r for _, r in rows[:TOP_K]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run_id", "lambda_acf", "lambda_freq", "lambda_motif", "lambda_cp", "lambda_per"]
        )
        for r in top:
            writer.writerow([
                r["run_id"],
                r["lambda_acf"],
                r["lambda_freq"],
                r["lambda_motif"],
                r["lambda_cp"],
                r["lambda_per"],
            ])

    print(f"Wrote top {len(top)} configs to {OUT_CSV}")


if __name__ == "__main__":
    main()
