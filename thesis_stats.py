#!/usr/bin/env python3
"""
thesis_stats.py — Statistical tests for MARL conveyor-belt thesis.

Runs Welch's t-test, Mann-Whitney U test, and Cohen's d effect size
on per-episode metrics from the 100-episode evaluation CSVs, following
the analysis plan in the thesis workflow (Step 19).

Usage:
    python thesis_stats.py

Expected files (adjust CSV_DIR if different):
  - Communication ablation (priority):
      results/eval_matd3_nocomm_s42.csv
      results/eval_matd3_comm1_s42.csv
      results/eval_matd3_comm3_s42.csv
  - Multi-seed MA-POCA (secondary):
      results/eval_mapoca_s42.csv
      results/eval_mapoca_s137.csv
      results/eval_mapoca_s256.csv
  - Heuristic baseline (reference):
      results/baseline_heuristic.csv

The script will skip any group whose CSVs are missing and report on
whatever it can find. Paste the printed output back to Claude so the
numbers can be folded into the thesis.

Dependencies: numpy, scipy, pandas.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_DIR = Path("results/")  # Adjust if your CSVs live elsewhere.

# Candidate column names to check in order. The script picks the first one
# that exists in the CSV header. Adjust here if your columns have other names.
ACCURACY_COLUMN_CANDIDATES = [
    "accuracy", "sort_accuracy", "SortAccuracy", "Accuracy", "mean_accuracy",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_accuracy(path: Path) -> np.ndarray | None:
    """Load the per-episode accuracy column from a CSV, or return None."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ACCURACY_COLUMN_CANDIDATES:
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    print(f"  WARN: {path.name} has no accuracy-like column. "
          f"Columns found: {list(df.columns)}")
    return None


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (unbiased, n-1)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled


def interpret_d(d: float) -> str:
    """Cohen's rule-of-thumb interpretation of the effect size."""
    ad = abs(d)
    if np.isnan(d):
        return "n/a"
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def shapiro_or_skip(x: np.ndarray, name: str) -> tuple[float, float] | None:
    """Shapiro-Wilk normality test, returns (W, p) or None if sample too small."""
    if len(x) < 3:
        return None
    try:
        return stats.shapiro(x)
    except Exception as exc:
        print(f"  WARN: Shapiro-Wilk failed on {name}: {exc}")
        return None


def compare_two(
    a_name: str, a: np.ndarray,
    b_name: str, b: np.ndarray,
) -> None:
    """Full comparison of two sample vectors."""
    print(f"\n{a_name} vs {b_name}")
    print("-" * 70)
    print(f"  n_{a_name} = {len(a)},  mean = {np.mean(a):.4f},  std = {np.std(a, ddof=1):.4f}")
    print(f"  n_{b_name} = {len(b)},  mean = {np.mean(b):.4f},  std = {np.std(b, ddof=1):.4f}")

    # Normality check — drives the choice between Welch (parametric) and
    # Mann-Whitney (rank-based).
    sa = shapiro_or_skip(a, a_name)
    sb = shapiro_or_skip(b, b_name)
    if sa is not None and sb is not None:
        print(f"  Shapiro-Wilk normality (p > 0.05 = normal):")
        print(f"    {a_name}: W = {sa.statistic:.4f}, p = {sa.pvalue:.4f}  "
              f"{'(normal)' if sa.pvalue > 0.05 else '(non-normal)'}")
        print(f"    {b_name}: W = {sb.statistic:.4f}, p = {sb.pvalue:.4f}  "
              f"{'(normal)' if sb.pvalue > 0.05 else '(non-normal)'}")

    # Welch's t-test (two-sided, unequal variances).
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  Welch's t-test:      t = {t_stat:+.4f},  p = {t_p:.4g}")

    # Mann-Whitney U (two-sided, non-parametric alternative).
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"  Mann-Whitney U:      U = {u_stat:.1f},  p = {u_p:.4g}")

    # Effect size.
    d = cohens_d(a, b)
    print(f"  Cohen's d:           d = {d:+.4f}  ({interpret_d(d)})")

    # Confidence interval for the mean difference (Welch-based, 95%).
    mean_diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    if se > 0:
        # Welch-Satterthwaite degrees of freedom
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        dfnum = (va / len(a) + vb / len(b)) ** 2
        dfden = ((va / len(a)) ** 2 / (len(a) - 1) +
                 (vb / len(b)) ** 2 / (len(b) - 1))
        df = dfnum / dfden if dfden > 0 else len(a) + len(b) - 2
        tcrit = stats.t.ppf(0.975, df)
        ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
        print(f"  Mean diff (A - B):   {mean_diff:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_DIR.exists():
        print(f"ERROR: directory {CSV_DIR} does not exist.")
        print("Edit CSV_DIR at the top of this script if your CSVs live elsewhere.")
        sys.exit(1)

    print(f"Reading CSVs from: {CSV_DIR.resolve()}")
    print("=" * 70)

    # --------- Communication ablation (priority, per workflow Step 19) ----
    print("\n[SECTION 1] COMMUNICATION ABLATION")
    print("=" * 70)
    nocomm = load_accuracy(CSV_DIR / "eval_matd3_nocomm_s42.csv")
    comm1  = load_accuracy(CSV_DIR / "eval_matd3_comm1_s42.csv")
    comm3  = load_accuracy(CSV_DIR / "eval_matd3_comm3_s42.csv")

    comm_runs = {"NoComm": nocomm, "Comm1": comm1, "Comm3": comm3}
    available = {k: v for k, v in comm_runs.items() if v is not None}
    if len(available) >= 2:
        names = list(available.keys())
        # NoComm vs Comm1, NoComm vs Comm3, Comm1 vs Comm3 (whichever pairs exist)
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                compare_two(a_name, available[a_name], b_name, available[b_name])
    else:
        print("  Need at least 2 of {nocomm, comm1, comm3} CSVs for this section.")
        print(f"  Found: {list(available.keys())}")

    # --------- Multi-seed MA-POCA ----------------------------------------
    print("\n\n[SECTION 2] MA-POCA MULTI-SEED VARIABILITY")
    print("=" * 70)
    ma_42  = load_accuracy(CSV_DIR / "eval_mapoca_s42.csv")
    ma_137 = load_accuracy(CSV_DIR / "eval_mapoca_s137.csv")
    ma_256 = load_accuracy(CSV_DIR / "eval_mapoca_s256.csv")
    seeds = {"s42": ma_42, "s137": ma_137, "s256": ma_256}
    available = {k: v for k, v in seeds.items() if v is not None}

    if len(available) >= 2:
        # Pairwise Welch / MW across seeds
        names = list(available.keys())
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                compare_two(f"MA-POCA_{a_name}", available[a_name],
                            f"MA-POCA_{b_name}", available[b_name])

        # Also report pooled mean and std for the thesis
        if len(available) >= 2:
            pooled = np.concatenate(list(available.values()))
            print(f"\n  Pooled across seeds: n = {len(pooled)}, "
                  f"mean = {np.mean(pooled):.4f}, "
                  f"std = {np.std(pooled, ddof=1):.4f}")
            # One-way ANOVA across seeds (sanity check for seed effect)
            if len(available) >= 3:
                f_stat, f_p = stats.f_oneway(*available.values())
                print(f"  One-way ANOVA across seeds: F = {f_stat:.4f}, p = {f_p:.4g}")
                print(f"    (p > 0.05 implies no detectable seed effect)")
    else:
        print("  Need at least 2 of the three MA-POCA seed CSVs for this section.")

    # --------- Heuristic baseline vs best learned ------------------------
    print("\n\n[SECTION 3] HEURISTIC vs BEST LEARNED POLICY")
    print("=" * 70)
    heuristic = load_accuracy(CSV_DIR / "baseline_heuristic.csv")
    if heuristic is not None:
        # Compare against whatever the best-available learned result is
        candidates = []
        if nocomm is not None: candidates.append(("MATD3_NoComm", nocomm))
        if ma_42 is not None:  candidates.append(("MA-POCA_s42", ma_42))
        if not candidates:
            print("  No learned-policy CSV available to compare against heuristic.")
        else:
            for name, arr in candidates:
                compare_two("Heuristic", heuristic, name, arr)
    else:
        print("  baseline_heuristic.csv not found.")

    print("\n" + "=" * 70)
    print("DONE. Paste this full output back to Claude for inclusion in the thesis.")
    print("=" * 70)


if __name__ == "__main__":
    main()