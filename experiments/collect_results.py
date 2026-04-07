"""
collect_results.py — Aggregate raw CSV into paper tables and figures.

Reads results/raw_results.csv produced by run_experiments.py and outputs:
  - Table I  (success rate & arrival rate) → printed to stdout
  - Table II (makespan & mean time)        → printed to stdout
  - Fig. 6   (success rate vs N)           → results/fig6_success_rate.png
  - Fig. 7   (arrival rate vs N)           → results/fig7_arrival_rate.png

Usage
-----
    cd /path/to/merry-go-round
    python experiments/collect_results.py
    python experiments/collect_results.py --csv results/raw_results.csv
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
ENV_ORDER    = ['free', 'circ15', 'rect15', 'swap']
METHOD_ORDER = ['mgr', 'clf_cbf', 'orca']
METHOD_LABEL = {'mgr': 'MGR', 'clf_cbf': 'CLF-CBF', 'orca': 'ORCA'}
METHOD_COLOR = {'mgr': 'steelblue', 'clf_cbf': 'tomato', 'orca': 'forestgreen'}
METHOD_MARKER = {'mgr': 'o', 'clf_cbf': 's', 'orca': '^'}


# ------------------------------------------------------------------
# Data loading & aggregation
# ------------------------------------------------------------------

def load_and_aggregate(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load raw CSV and return grouped statistics per (env, N, method).

    Columns in output:
        env, N, method,
        success_mean, success_std,
        arrival_mean, arrival_std,
        makespan_mean, makespan_std,   # NaN where no run fully succeeded
        meantime_mean, meantime_std,
        n_runs                         # should be N_INSTANCES = 20
    """
    df = pd.read_csv(csv_path)

    # Ensure numeric types (CSV may store None as empty string)
    for col in ['makespan', 'mean_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg = (
        df.groupby(['env', 'N', 'method'])
        .agg(
            success_mean  = ('success_rate', 'mean'),
            success_std   = ('success_rate', 'std'),
            arrival_mean  = ('arrival_rate', 'mean'),
            arrival_std   = ('arrival_rate', 'std'),
            makespan_mean = ('makespan',     'mean'),
            makespan_std  = ('makespan',     'std'),
            meantime_mean = ('mean_time',    'mean'),
            meantime_std  = ('mean_time',    'std'),
            n_runs        = ('success_rate', 'count'),
        )
        .reset_index()
    )
    return agg


# ------------------------------------------------------------------
# Table I — Success rate & arrival rate
# ------------------------------------------------------------------

def print_table_i(agg: pd.DataFrame) -> None:
    """Print Table I in a readable format matching the paper layout."""
    print("=" * 80)
    print("TABLE I — Success Rate and Arrival Rate (mean over 20 instances)")
    print("=" * 80)

    header = f"{'Env':<8} {'N':>4}  "
    for m in METHOD_ORDER:
        header += f"  {METHOD_LABEL[m]+'-Suc':>10}"
    header += "  |"
    for m in METHOD_ORDER:
        header += f"  {METHOD_LABEL[m]+'-Arr':>10}"
    print(header)
    print("-" * 80)

    for env in ENV_ORDER:
        sub = agg[agg.env == env].sort_values('N')
        if sub.empty:
            continue
        for _, row_all in sub[sub.method == METHOD_ORDER[0]].iterrows():
            N = int(row_all.N)
            line = f"{env:<8} {N:>4}  "
            for m in METHOD_ORDER:
                r = agg[(agg.env == env) & (agg.N == N) & (agg.method == m)]
                val = f"{r.success_mean.values[0]:.2f}" if not r.empty else "  —  "
                line += f"  {val:>10}"
            line += "  |"
            for m in METHOD_ORDER:
                r = agg[(agg.env == env) & (agg.N == N) & (agg.method == m)]
                val = f"{r.arrival_mean.values[0]:.2f}" if not r.empty else "  —  "
                line += f"  {val:>10}"
            print(line)
        print()

    print()


# ------------------------------------------------------------------
# Table II — Makespan & mean time
# ------------------------------------------------------------------

def print_table_ii(agg: pd.DataFrame) -> None:
    """Print Table II in a readable format matching the paper layout."""
    print("=" * 80)
    print("TABLE II — Makespan and Mean Time (mean ± std, seconds)")
    print("         (— = no instances fully succeeded)")
    print("=" * 80)

    col_w = 18
    header = f"{'Env':<8} {'N':>4}  "
    for m in METHOD_ORDER:
        header += f"  {(METHOD_LABEL[m]+' Makespan'):<{col_w}}"
    header += "  |"
    for m in METHOD_ORDER:
        header += f"  {(METHOD_LABEL[m]+' MeanTime'):<{col_w}}"
    print(header)
    print("-" * 100)

    def fmt(mean, std):
        if pd.isna(mean):
            return "—"
        return f"{mean:.1f}±{std:.1f}s"

    for env in ENV_ORDER:
        sub = agg[agg.env == env].sort_values('N')
        if sub.empty:
            continue
        for _, _ in sub[sub.method == METHOD_ORDER[0]].iterrows():
            N = int(_.N)
            line = f"{env:<8} {N:>4}  "
            for m in METHOD_ORDER:
                r = agg[(agg.env == env) & (agg.N == N) & (agg.method == m)]
                val = fmt(r.makespan_mean.values[0], r.makespan_std.values[0]) if not r.empty else "—"
                line += f"  {val:<{col_w}}"
            line += "  |"
            for m in METHOD_ORDER:
                r = agg[(agg.env == env) & (agg.N == N) & (agg.method == m)]
                val = fmt(r.meantime_mean.values[0], r.meantime_std.values[0]) if not r.empty else "—"
                line += f"  {val:<{col_w}}"
            print(line)
        print()

    print()


# ------------------------------------------------------------------
# Fig. 6 — Success rate vs N
# ------------------------------------------------------------------

def plot_success_rate(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Reproduce paper Fig. 6: success rate vs N, 4 subplots."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    fig.suptitle('Fig. 6 — Success Rate vs N Robots', fontsize=12)

    for ax, env in zip(axes, ENV_ORDER):
        for method in METHOD_ORDER:
            sub = agg[(agg.env == env) & (agg.method == method)].sort_values('N')
            if sub.empty:
                continue
            color  = METHOD_COLOR[method]
            marker = METHOD_MARKER[method]
            label  = METHOD_LABEL[method]
            lo = (sub.success_mean - sub.success_std).clip(0.0, 1.0)
            hi = (sub.success_mean + sub.success_std).clip(0.0, 1.0)
            ax.plot(sub.N, sub.success_mean, color=color, marker=marker,
                    label=label, linewidth=1.5, markersize=5)
            ax.fill_between(sub.N, lo, hi, alpha=0.2, color=color)

        ax.set_title(env.upper(), fontsize=10)
        ax.set_xlabel('N robots', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel('Success rate', fontsize=9)
    axes[-1].legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Fig. 7 — Arrival rate vs N
# ------------------------------------------------------------------

def plot_arrival_rate(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Reproduce paper Fig. 7: arrival rate vs N, 4 subplots."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    fig.suptitle('Fig. 7 — Arrival Rate vs N Robots', fontsize=12)

    for ax, env in zip(axes, ENV_ORDER):
        for method in METHOD_ORDER:
            sub = agg[(agg.env == env) & (agg.method == method)].sort_values('N')
            if sub.empty:
                continue
            color  = METHOD_COLOR[method]
            marker = METHOD_MARKER[method]
            label  = METHOD_LABEL[method]
            lo = (sub.arrival_mean - sub.arrival_std).clip(0.0, 1.0)
            hi = (sub.arrival_mean + sub.arrival_std).clip(0.0, 1.0)
            ax.plot(sub.N, sub.arrival_mean, color=color, marker=marker,
                    label=label, linewidth=1.5, markersize=5)
            ax.fill_between(sub.N, lo, hi, alpha=0.2, color=color)

        ax.set_title(env.upper(), fontsize=10)
        ax.set_xlabel('N robots', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel('Arrival rate', fontsize=9)
    axes[-1].legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results.')
    parser.add_argument('--csv', default='results/raw_results.csv',
                        help='Path to raw_results.csv (default: results/raw_results.csv)')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run experiments/run_experiments.py first.")
        sys.exit(1)

    print(f"Loading {csv_path} …")
    agg = load_and_aggregate(csv_path)

    n_rows = len(pd.read_csv(csv_path))
    print(f"Total runs in CSV: {n_rows}")
    print()

    print_table_i(agg)
    print_table_ii(agg)

    results_dir = pathlib.Path('results')
    results_dir.mkdir(exist_ok=True)
    plot_success_rate(agg, results_dir / 'fig6_success_rate.png')
    plot_arrival_rate(agg, results_dir / 'fig7_arrival_rate.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
