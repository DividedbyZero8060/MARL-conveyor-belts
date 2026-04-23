"""
Re-compute message analysis metrics from previously saved per-agent CSVs.

Addresses the MI over-counting bug from the original analyze_messages.py:
the original summed per-feature MI values, which over-counts when features
are correlated (as PCA showed comm3 messages effectively are).

This script:
  - Reads <run_id>_agent{N}_messages.csv files from a directory
  - Recomputes joint MI via PCA-to-1D projection
  - Recomputes PCA and entropy (unchanged math, just reads from CSV)
  - Writes a corrected summary

Usage:
    python scripts/analyze_messages_from_csv.py \\
        --run-id=ablation_comm3_level2 \\
        --bandwidth=3
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


NUM_AGENTS = 3
MAX_MI_3CLASS = np.log(3)  # ≈ 1.099 nats — ceiling for MI with 3-class label


def load_agent_csv(csv_path: Path, bandwidth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load (messages, types) arrays from a per-agent CSV."""
    messages = []
    types = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        msg_cols = [i for i, h in enumerate(header) if h.startswith("msg_")]
        type_col = header.index("package_type")

        if len(msg_cols) != bandwidth:
            raise ValueError(
                f"CSV {csv_path}: expected {bandwidth} msg_ columns, "
                f"found {len(msg_cols)}"
            )

        for row in reader:
            msg = [float(row[c]) for c in msg_cols]
            t = int(row[type_col])
            messages.append(msg)
            types.append(t)

    return np.asarray(messages, dtype=np.float32), np.asarray(types, dtype=np.int32)


def compute_joint_mi(
    messages: np.ndarray,
    types: np.ndarray,
    n_neighbors: int = 3,
) -> float:
    """
    Joint MI(message_vector, type_label) via PCA-to-1D.

    Rationale: mutual_info_classif returns per-feature MI. Summing them
    over-counts when features are correlated. Projecting to the top PCA
    component gives a 1-D signal that captures most of the variance
    (PCA PC1 >= 80% in all three agents' comm3 results), and a single-
    feature MI estimate is unbiased by correlation.

    Bounded above by log(n_classes). For 3-class labels: ≈ 1.099 nats.
    """
    if not _HAS_SKLEARN:
        return 0.0

    mask = types >= 0
    if mask.sum() < 10:
        return 0.0

    X = messages[mask]
    y = types[mask]

    if X.shape[1] > 1:
        pca = PCA(n_components=1)
        X_1d = pca.fit_transform(X)
    else:
        X_1d = X

    mi = mutual_info_classif(X_1d, y, n_neighbors=n_neighbors, random_state=0)
    return float(mi[0])


def compute_pca(messages: np.ndarray, bandwidth: int) -> Dict[str, float]:
    """Variance explained by each component. Unchanged from the original."""
    if not _HAS_SKLEARN or bandwidth < 2 or len(messages) < bandwidth:
        return {"PC1": 1.0}
    if bandwidth == 3:
        pca = PCA(n_components=3)
        pca.fit(messages)
        v = pca.explained_variance_ratio_
        return {"PC1": float(v[0]), "PC2": float(v[1]), "PC3": float(v[2])}
    return {"PC1": 1.0}


def compute_entropy_ratio(messages: np.ndarray, n_bins: int = 20) -> float:
    """Entropy ratio per-dim, averaged. 1.0 = uniform, <1 = structured."""
    if len(messages) < n_bins:
        return 1.0

    total = 0.0
    uniform = np.log(n_bins)
    for dim in range(messages.shape[1]):
        hist, _ = np.histogram(messages[:, dim], bins=n_bins, range=(0, 1))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        total += -np.sum(probs * np.log(probs))
    return float((total / messages.shape[1]) / uniform)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-analyse message CSVs with corrected joint MI.",
    )
    p.add_argument("--run-id", type=str, required=True,
                   help="Run ID used to name the CSV files (e.g. ablation_comm3_level2).")
    p.add_argument("--bandwidth", type=int, required=True, choices=[1, 3],
                   help="Message bandwidth used during training.")
    p.add_argument("--input-dir", type=str, default="results/msg_analysis",
                   help="Directory containing the per-agent CSVs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HAS_SKLEARN:
        print("ERROR: scikit-learn is required. pip install scikit-learn")
        sys.exit(1)

    input_dir = Path(args.input_dir)

    print("=" * 60)
    print(f"Re-analysis: {args.run_id} (bandwidth={args.bandwidth})")
    print("=" * 60)

    mi_results: Dict[int, float] = {}
    pca_results: Dict[int, Dict[str, float]] = {}
    ent_results: Dict[int, float] = {}
    sample_counts: Dict[int, Tuple[int, int]] = {}

    for i in range(NUM_AGENTS):
        csv_path = input_dir / f"{args.run_id}_agent{i}_messages.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}")
            continue

        messages, types = load_agent_csv(csv_path, args.bandwidth)
        n_total = len(messages)
        n_valid = int((types >= 0).sum())
        sample_counts[i] = (n_total, n_valid)

        mi_results[i] = compute_joint_mi(messages, types)
        pca_results[i] = compute_pca(messages, args.bandwidth)
        ent_results[i] = compute_entropy_ratio(messages)

    # ---- Print corrected summary ----
    print()
    print("Sample counts:")
    for i in range(NUM_AGENTS):
        if i in sample_counts:
            tot, val = sample_counts[i]
            print(f"  Agent {i}: {tot} total, {val} with non-empty slot 0")

    print()
    print(f"--- Joint MI(message, package_type)  [ceiling: log(3) = {MAX_MI_3CLASS:.3f} nats] ---")
    for i in range(NUM_AGENTS):
        if i not in mi_results:
            continue
        val = mi_results[i]
        frac = val / MAX_MI_3CLASS
        if frac > 0.5:
            tag = "STRONG"
        elif frac > 0.1:
            tag = "WEAK"
        else:
            tag = "NOISE"
        print(f"  Agent {i}: MI = {val:.4f} nats  ({100*frac:5.1f}% of ceiling)  [{tag}]")

    print()
    print("--- PCA on messages ---")
    for i in range(NUM_AGENTS):
        if i not in pca_results or "PC3" not in pca_results[i]:
            continue
        pc = pca_results[i]
        print(f"  Agent {i}: PC1={pc['PC1']:.3f}  PC2={pc['PC2']:.3f}  PC3={pc['PC3']:.3f}")
        if pc["PC1"] > 0.80:
            print(f"           → effective dim ≈ 1, bandwidth {args.bandwidth} mostly wasted")

    print()
    print("--- Entropy ratio (1.0 = uniform, <1 = structured) ---")
    for i in range(NUM_AGENTS):
        if i not in ent_results:
            continue
        val = ent_results[i]
        tag = "STRUCTURED" if val < 0.9 else "UNIFORM"
        print(f"  Agent {i}: entropy ratio = {val:.4f}  [{tag}]")

    # ---- Save corrected summary ----
    summary_path = input_dir / f"{args.run_id}_summary_corrected.txt"
    with open(summary_path, "w") as f:
        f.write(f"Corrected message analysis: {args.run_id}\n")
        f.write(f"Bandwidth: {args.bandwidth}\n")
        f.write(f"MI ceiling (log(3)): {MAX_MI_3CLASS:.4f} nats\n\n")
        f.write("Joint MI(message, package_type):\n")
        for i in range(NUM_AGENTS):
            if i in mi_results:
                val = mi_results[i]
                f.write(f"  Agent {i}: {val:.4f} nats  ({100*val/MAX_MI_3CLASS:.1f}% of ceiling)\n")
        f.write("\nPCA:\n")
        for i in range(NUM_AGENTS):
            if i in pca_results and "PC3" in pca_results[i]:
                pc = pca_results[i]
                f.write(f"  Agent {i}: PC1={pc['PC1']:.3f}  PC2={pc['PC2']:.3f}  PC3={pc['PC3']:.3f}\n")
        f.write("\nEntropy ratio:\n")
        for i in range(NUM_AGENTS):
            if i in ent_results:
                f.write(f"  Agent {i}: {ent_results[i]:.4f}\n")
    print()
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()