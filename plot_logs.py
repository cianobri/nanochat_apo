#!/usr/bin/env python3

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_DIR = Path("logs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TRAIN_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?\|\s+loss:\s+([0-9.eE+-]+)"
)

VAL_RE = re.compile(
    r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.eE+-]+)"
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = path.stem
    train_rows = []
    val_rows = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = strip_ansi(line)

            m = TRAIN_RE.search(line)
            if m:
                step = int(m.group(1))
                total_steps = int(m.group(2))
                loss = float(m.group(3))
                train_rows.append(
                    {
                        "method": method,
                        "step": step,
                        "total_steps": total_steps,
                        "train_loss": loss,
                        "log_file": str(path),
                    }
                )
                continue

            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                val_bpb = float(m.group(2))
                val_rows.append(
                    {
                        "method": method,
                        "step": step,
                        "val_bpb": val_bpb,
                        "log_file": str(path),
                    }
                )

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def main() -> None:
    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No .log files found in {LOG_DIR.resolve()}")

    train_dfs = []
    val_dfs = []

    for path in log_files:
        train_df, val_df = parse_log(path)
        if not train_df.empty:
            train_dfs.append(train_df)
        if not val_df.empty:
            val_dfs.append(val_df)

    if not train_dfs:
        raise RuntimeError("No training loss lines found in logs.")
    if not val_dfs:
        raise RuntimeError("No validation BPB lines found in logs.")

    train = pd.concat(train_dfs, ignore_index=True)
    val = pd.concat(val_dfs, ignore_index=True)

    train.to_csv(OUT_DIR / "train_loss.csv", index=False)
    val.to_csv(OUT_DIR / "val_bpb.csv", index=False)

    # Plot 1: train loss vs step
    plt.figure(figsize=(10, 6))
    for method, df in train.groupby("method"):
        df = df.sort_values("step")
        plt.plot(df["step"], df["train_loss"], label=method)

    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss vs step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_loss_vs_step.png", dpi=200)
    plt.close()

    # Plot 2: validation BPB vs step
    plt.figure(figsize=(10, 6))
    for method, df in val.groupby("method"):
        df = df.sort_values("step")
        plt.plot(df["step"], df["val_bpb"], marker="o", label=method)

    plt.xlabel("Step")
    plt.ylabel("Validation BPB")
    plt.title("Validation BPB vs step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "val_bpb_vs_step.png", dpi=200)
    plt.close()

    print(f"Parsed {len(log_files)} log files.")
    print(f"Wrote: {OUT_DIR / 'train_loss.csv'}")
    print(f"Wrote: {OUT_DIR / 'val_bpb.csv'}")
    print(f"Wrote: {OUT_DIR / 'train_loss_vs_step.png'}")
    print(f"Wrote: {OUT_DIR / 'val_bpb_vs_step.png'}")

    print("\nBest validation BPB per method:")
    best = val.loc[val.groupby("method")["val_bpb"].idxmin()]
    best = best.sort_values("val_bpb")
    print(best[["method", "step", "val_bpb"]].to_string(index=False))


if __name__ == "__main__":
    main()