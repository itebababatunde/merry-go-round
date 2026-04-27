"""
make_tables.py — Render Table I and Table II from the paper as PNG images.

Output:
    results/table1_success_arrival.png
    results/table2_makespan_meantime.png

Usage:
    python3 experiments/make_tables.py
"""

import pathlib
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH   = pathlib.Path('results/raw_results.csv')
OUT_DIR    = pathlib.Path('results')

ENV_ORDER    = ['free', 'circ15', 'rect15', 'swap']
ENV_LABEL    = {'free': 'Free', 'circ15': 'Circ-15%', 'rect15': 'Rect-15%', 'swap': 'Swap'}
METHOD_ORDER = ['mgr', 'clf_cbf', 'orca']
METHOD_LABEL = {'mgr': 'MGR', 'clf_cbf': 'CLF-CBF', 'orca': 'ORCA'}

# Visual palette — one colour per method (light fill for table cells)
METHOD_COLOR = {'mgr': '#dce8f7', 'clf_cbf': '#fde8e8', 'orca': '#e2f4e7'}
HEADER_COLOR = '#2c3e50'
ALT_ROW      = '#f5f6f7'   # alternating row tint
BORDER       = '#bdc3c7'


# ── Data loading ─────────────────────────────────────────────────────────────

def load_agg(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['makespan', 'mean_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg = (
        df.groupby(['env', 'N', 'method'])
        .agg(
            success_mean  = ('success_rate', 'mean'),
            arrival_mean  = ('arrival_rate', 'mean'),
            makespan_mean = ('makespan',     'mean'),
            makespan_std  = ('makespan',     'std'),
            meantime_mean = ('mean_time',    'mean'),
            meantime_std  = ('mean_time',    'std'),
            n_runs        = ('success_rate', 'count'),
        )
        .reset_index()
    )
    return agg


def get(agg, env, N, method, col):
    r = agg[(agg.env == env) & (agg.N == N) & (agg.method == method)]
    return r[col].values[0] if not r.empty else np.nan


# ── Table builder ─────────────────────────────────────────────────────────────

def _pct(v):
    """Format a [0,1] rate as integer percent string, or '—' if NaN."""
    if np.isnan(v):
        return '—'
    return f'{round(v * 100):d}%'


def _time(mean, std):
    """Format mean±std seconds, or '—' if NaN."""
    if np.isnan(mean):
        return '—'
    if np.isnan(std):
        return f'{mean:.1f}s'
    return f'{mean:.1f}±{std:.1f}s'


def _draw_table(ax, col_labels, row_groups, cell_data, cell_colors,
                col_widths, row_height=0.12, header_height=0.14,
                subheader_height=0.10):
    """
    Draw a styled table onto `ax` (which should have axis('off')).

    Parameters
    ----------
    col_labels   : list of str  — top header labels
    row_groups   : list of (group_label, [row_label, ...]) — left side
    cell_data    : 2-D list [row][col] of str
    cell_colors  : 2-D list [row][col] of colour strings
    col_widths   : list of float, len = len(col_labels) + 1  (incl. left label col)
    """
    ax.set_xlim(0, sum(col_widths))
    n_header_rows = 1
    total_data_rows = sum(len(rows) for _, rows in row_groups)
    n_group_labels  = len(row_groups)
    total_height = (header_height
                    + n_group_labels * subheader_height
                    + total_data_rows * row_height)
    ax.set_ylim(0, total_height)
    ax.invert_yaxis()

    def cell(x, y, w, h, text, facecolor, fontsize=8.5, bold=False,
             color='black', align='center', valign='center'):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='square,pad=0',
            facecolor=facecolor, edgecolor=BORDER, linewidth=0.5,
        )
        ax.add_patch(rect)
        ha = align
        ax.text(x + w / 2 if align == 'center' else x + 0.01 * w,
                y + h / 2,
                text,
                ha=ha, va=valign,
                fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                color=color,
                clip_on=True)

    # ── Header row ──────────────────────────────────────────────────────────
    x = 0.0
    y = 0.0
    # Empty top-left corner
    cell(x, y, col_widths[0], header_height,
         '', HEADER_COLOR, color='white', bold=True)
    x += col_widths[0]
    for i, lbl in enumerate(col_labels):
        w = col_widths[i + 1]
        cell(x, y, w, header_height,
             lbl, HEADER_COLOR, fontsize=8.5, bold=True, color='white')
        x += w

    # ── Data rows ────────────────────────────────────────────────────────────
    y = header_height
    data_row_idx = 0

    for g_idx, (group_label, sub_rows) in enumerate(row_groups):
        # Group sub-header (env label spanning full width)
        total_w = sum(col_widths)
        cell(0, y, total_w, subheader_height,
             group_label, '#dde3ea', fontsize=8.5, bold=True,
             align='left', color='#1a252f')
        y += subheader_height

        for r_idx, row_label in enumerate(sub_rows):
            alt = ALT_ROW if r_idx % 2 == 1 else 'white'

            # Left label column
            cell(0, y, col_widths[0], row_height,
                 row_label, alt, fontsize=8, align='left', color='#2c3e50')

            x = col_widths[0]
            for c_idx in range(len(col_labels)):
                w   = col_widths[c_idx + 1]
                txt = cell_data[data_row_idx][c_idx]
                fc  = cell_colors[data_row_idx][c_idx]
                if fc is None:
                    fc = alt
                cell(x, y, w, row_height, txt, fc, fontsize=8.5)
                x += w

            y += row_height
            data_row_idx += 1


# ── Table I ──────────────────────────────────────────────────────────────────

def make_table_i(agg, out_path):
    all_Ns = {env: sorted(agg[agg.env == env]['N'].unique()) for env in ENV_ORDER}

    # Build col labels: MGR / CLF-CBF / ORCA  ×  Suc / Arr
    col_labels = []
    for m in METHOD_ORDER:
        col_labels.append(f'{METHOD_LABEL[m]}\nSuccess')
        col_labels.append(f'{METHOD_LABEL[m]}\nArrival')

    # Row groups
    row_groups = []
    cell_data  = []
    cell_colors = []

    for env in ENV_ORDER:
        ns = all_Ns[env]
        sub_rows = [f'  N = {N}' for N in ns]
        row_groups.append((f'{ENV_LABEL[env]}', sub_rows))
        for N in ns:
            row = []
            colors = []
            for m in METHOD_ORDER:
                suc = get(agg, env, N, m, 'success_mean')
                arr = get(agg, env, N, m, 'arrival_mean')
                row.append(_pct(suc))
                row.append(_pct(arr))
                # Colour cells by method tint; highlight 100% in stronger shade
                mc = METHOD_COLOR[m]
                colors.append(mc if not np.isnan(suc) else '#f0f0f0')
                colors.append(mc if not np.isnan(arr) else '#f0f0f0')
            cell_data.append(row)
            cell_colors.append(colors)

    n_cols = len(col_labels)
    col_widths = [0.80] + [0.70] * n_cols   # left label + data cols

    fig_w = sum(col_widths) * 1.5
    fig_h = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    _draw_table(ax, col_labels, row_groups, cell_data, cell_colors,
                col_widths, row_height=0.13, header_height=0.18,
                subheader_height=0.11)

    ax.set_title('Table I — Success Rate and Arrival Rate (mean over 20 instances)',
                 fontsize=10, fontweight='bold', pad=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Table II ─────────────────────────────────────────────────────────────────

def make_table_ii(agg, out_path):
    all_Ns = {env: sorted(agg[agg.env == env]['N'].unique()) for env in ENV_ORDER}

    col_labels = []
    for m in METHOD_ORDER:
        col_labels.append(f'{METHOD_LABEL[m]}\nMakespan')
        col_labels.append(f'{METHOD_LABEL[m]}\nMean Time')

    row_groups  = []
    cell_data   = []
    cell_colors = []

    for env in ENV_ORDER:
        ns = all_Ns[env]
        sub_rows = [f'  N = {N}' for N in ns]
        row_groups.append((f'{ENV_LABEL[env]}', sub_rows))
        for N in ns:
            row = []
            colors = []
            for m in METHOD_ORDER:
                ms_m = get(agg, env, N, m, 'makespan_mean')
                ms_s = get(agg, env, N, m, 'makespan_std')
                mt_m = get(agg, env, N, m, 'meantime_mean')
                mt_s = get(agg, env, N, m, 'meantime_std')
                row.append(_time(ms_m, ms_s))
                row.append(_time(mt_m, mt_s))
                mc = METHOD_COLOR[m]
                colors.append(mc if not np.isnan(ms_m) else '#f0f0f0')
                colors.append(mc if not np.isnan(mt_m) else '#f0f0f0')
            cell_data.append(row)
            cell_colors.append(colors)

    n_cols = len(col_labels)
    col_widths = [0.80] + [1.05] * n_cols

    fig_w = sum(col_widths) * 1.5
    fig_h = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    _draw_table(ax, col_labels, row_groups, cell_data, cell_colors,
                col_widths, row_height=0.13, header_height=0.18,
                subheader_height=0.11)

    ax.set_title(
        'Table II — Makespan and Mean Time (mean ± std, seconds)\n'
        '— = no instances fully succeeded',
        fontsize=10, fontweight='bold', pad=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not CSV_PATH.exists():
        print(f'ERROR: {CSV_PATH} not found. Run experiments/run_experiments.py first.')
        sys.exit(1)

    agg = load_agg(CSV_PATH)
    make_table_i(agg,  OUT_DIR / 'table1_success_arrival.png')
    make_table_ii(agg, OUT_DIR / 'table2_makespan_meantime.png')
    print('Done.')
