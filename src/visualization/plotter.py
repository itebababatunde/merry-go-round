"""
plotter.py — Publication-quality Figs. 6 & 7 for the IEEE report.

Reads results/raw_results.csv (produced by experiments/run_experiments.py)
and saves Figs. 6 and 7 in PNG or PDF format, sized for an IEEE double-column
paper (7.16 in wide).

Usage
-----
    # From the command line
    python3 src/visualization/plotter.py
    python3 src/visualization/plotter.py --csv results/raw_results.csv --fmt pdf

    # From Python
    from src.visualization.plotter import plot_figures
    plot_figures(csv_path='results/raw_results.csv', fmt='pdf')
"""

import argparse
import pathlib
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ------------------------------------------------------------------
# Style constants
# ------------------------------------------------------------------
ENV_ORDER    = ['free', 'circ15', 'rect15', 'swap']
ENV_LABEL    = {'free': 'Free', 'circ15': 'Circ-15%',
                'rect15': 'Rect-15%', 'swap': 'Swap'}
METHOD_ORDER = ['mgr', 'clf_cbf', 'orca']
METHOD_LABEL = {'mgr': 'MGR', 'clf_cbf': 'CLF-CBF', 'orca': 'ORCA'}
METHOD_COLOR = {'mgr': '#3a86ff', 'clf_cbf': '#e63946', 'orca': '#2dc653'}
METHOD_DASH  = {'mgr': '-',       'clf_cbf': '--',      'orca': '-.'}
METHOD_MARK  = {'mgr': 'o',       'clf_cbf': 's',       'orca': '^'}

# IEEE double-column width = 7.16 in; row height tuned for readability
_FIG_W = 7.16
_FIG_H = 2.8

# rcParams for IEEE-style fonts
_RC = {
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'legend.fontsize':  6,
    'lines.linewidth':  1.2,
    'lines.markersize': 4,
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load raw CSV and return per-(env,N,method) aggregated stats."""
    df = pd.read_csv(csv_path)
    for col in ['makespan', 'mean_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg = (
        df.groupby(['env', 'N', 'method'])
        .agg(
            success_mean = ('success_rate', 'mean'),
            success_std  = ('success_rate', 'std'),
            arrival_mean = ('arrival_rate', 'mean'),
            arrival_std  = ('arrival_rate', 'std'),
            n_runs       = ('success_rate', 'count'),
        )
        .reset_index()
    )
    return agg


# ------------------------------------------------------------------
# Generic 4-subplot rate figure
# ------------------------------------------------------------------

def _plot_rate_figure(
    agg: pd.DataFrame,
    rate_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    out_path: pathlib.Path,
    dpi: int,
) -> None:
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 4, figsize=(_FIG_W, _FIG_H), sharey=True)
        fig.suptitle(title, fontsize=8, y=1.01)

        for ax, env in zip(axes, ENV_ORDER):
            has_data = False
            for method in METHOD_ORDER:
                sub = agg[(agg.env == env) & (agg.method == method)].sort_values('N')
                if sub.empty:
                    continue
                has_data = True
                color = METHOD_COLOR[method]
                dash  = METHOD_DASH[method]
                mark  = METHOD_MARK[method]
                label = METHOD_LABEL[method]
                mean  = sub[rate_col].values
                std   = sub[std_col].fillna(0).values
                xs    = sub['N'].values

                ax.plot(xs, mean, linestyle=dash, marker=mark,
                        color=color, label=label)
                ax.fill_between(xs,
                                np.clip(mean - std, 0.0, 1.0),
                                np.clip(mean + std, 0.0, 1.0),
                                alpha=0.15, color=color)

            ax.set_title(ENV_LABEL.get(env, env))
            ax.set_xlabel('N robots')
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.7)

            if has_data:
                ax.legend(loc='lower left', framealpha=0.8)

        axes[0].set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
        plt.close()
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def plot_figures(
    csv_path: str = 'results/raw_results.csv',
    output_dir: str = 'results',
    fmt: str = 'png',
    dpi: int = 200,
) -> None:
    """
    Generate Figs. 6 & 7 from raw_results.csv.

    Parameters
    ----------
    csv_path   : path to the CSV produced by run_experiments.py
    output_dir : directory to save output files
    fmt        : 'png' or 'pdf'
    dpi        : resolution for PNG output (ignored for PDF)
    """
    csv_path   = pathlib.Path(csv_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run `python3 experiments/run_experiments.py` first.")
        return

    print(f"Loading {csv_path} …")
    agg = _load(csv_path)
    n_runs = len(pd.read_csv(csv_path))
    print(f"Total runs: {n_runs}  |  "
          f"(env,N,method) groups: {len(agg)}")

    _plot_rate_figure(
        agg,
        rate_col='success_mean', std_col='success_std',
        ylabel='Success rate',
        title='Fig. 6 — Success Rate vs Number of Robots',
        out_path=output_dir / f'fig6_success_rate.{fmt}',
        dpi=dpi,
    )

    _plot_rate_figure(
        agg,
        rate_col='arrival_mean', std_col='arrival_std',
        ylabel='Arrival rate',
        title='Fig. 7 — Arrival Rate vs Number of Robots',
        out_path=output_dir / f'fig7_arrival_rate.{fmt}',
        dpi=dpi,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figs 6 & 7.')
    parser.add_argument('--csv', default='results/raw_results.csv')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--fmt', default='png', choices=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    plot_figures(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fmt=args.fmt,
        dpi=args.dpi,
    )
