"""
Produce cleaner histogram plots of message-dimension distributions per agent,
colored by package type. Reads the CSVs produced by analyze_messages.py.

Much easier to read than the 2D scatter plots when channels saturate.

Usage:
    python scripts/plot_message_histograms.py \\
        --run-id=ablation_comm3_level2_updated \\
        --bandwidth=3
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


NUM_AGENTS = 3
TYPE_LABELS = {0: "DestA", 1: "DestB", 2: "DestC"}
TYPE_COLORS = {0: "steelblue", 1: "darkorange", 2: "seagreen"}


def load_agent_csv(csv_path: Path, bandwidth: int) -> Tuple[np.ndarray, np.ndarray]:
    messages = []
    types = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        msg_cols = [i for i, h in enumerate(header) if h.startswith("msg_")]
        type_col = header.index("package_type")

        for row in reader:
            msg = [float(row[c]) for c in msg_cols]
            t = int(row[type_col])
            messages.append(msg)
            types.append(t)

    return np.asarray(messages, dtype=np.float32), np.asarray(types, dtype=np.int32)


def plot_agent_histograms(
    agent_idx: int,
    messages: np.ndarray,
    types: np.ndarray,
    bandwidth: int,
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Produce one figure per agent with (bandwidth) subplots — one histogram
    per message dimension, stacked and colored by package type.
    """
    if not _HAS_MPL:
        return

    mask = types >= 0
    messages = messages[mask]
    types = types[mask]
    if len(messages) < 10:
        print(f"  Agent {agent_idx}: not enough valid samples, skipping.")
        return

    n_bins = 40
    fig, axes = plt.subplots(1, bandwidth, figsize=(5 * bandwidth, 4.5), sharey=True)
    if bandwidth == 1:
        axes = [axes]

    # Detect which types are actually present for the legend
    present_types = sorted(set(int(t) for t in np.unique(types)))

    for dim in range(bandwidth):
        ax = axes[dim]
        for t in present_types:
            type_mask = types == t
            if type_mask.sum() == 0:
                continue
            ax.hist(
                messages[type_mask, dim],
                bins=n_bins, range=(0, 1),
                alpha=0.6,
                color=TYPE_COLORS.get(t, "gray"),
                label=f"{TYPE_LABELS.get(t, f'Type {t}')} (n={int(type_mask.sum())})",
                edgecolor="black",
                linewidth=0.3,
            )
        ax.set_xlabel(f"message[{dim}] value")
        if dim == 0:
            ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.set_title(f"Agent {agent_idx}, dim {dim}")
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{run_id}: Agent {agent_idx} message distributions by package type", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{run_id}_agent{agent_idx}_histograms.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce per-dimension histograms from message CSVs.")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--bandwidth", type=int, required=True, choices=[1, 3])
    p.add_argument("--input-dir", type=str, default="results/msg_analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not _HAS_MPL:
        print("ERROR: matplotlib not installed. pip install matplotlib")
        sys.exit(1)

    input_dir = Path(args.input_dir)

    print("=" * 60)
    print(f"Histogram plots: {args.run_id} (bandwidth={args.bandwidth})")
    print("=" * 60)

    for i in range(NUM_AGENTS):
        csv_path = input_dir / f"{args.run_id}_agent{i}_messages.csv"
        if not csv_path.exists():
            print(f"  WARNING: missing {csv_path}, skipping agent {i}")
            continue

        messages, types = load_agent_csv(csv_path, args.bandwidth)
        print(f"  Agent {i}: {len(messages)} samples, {int((types >= 0).sum())} valid")
        plot_agent_histograms(i, messages, types, args.bandwidth, input_dir, args.run_id)

    print("Done.")


if __name__ == "__main__":
    main()